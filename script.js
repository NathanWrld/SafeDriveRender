import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm'
import { startDetection, stopDetection, toggleNightMode } from './detection.js';

const supabaseUrl = 'https://roogjmgxghbuiogpcswy.supabase.co'
const supabaseKey = 'sb_publishable_RTN2PXvdWOQFfUySAaTa_g_LLe-T_NU'
const supabase = createClient(supabaseUrl, supabaseKey)

// -------------------- VARIABLES GLOBALES --------------------
let sessionId = null;
let maxSessionRisk = 0; // 0:Normal, 1:Leve, 2:Moderado, 3:Alto
const videoElement = document.querySelector('.input_video');
const canvasElement = document.querySelector('.output_canvas');
const estado = document.getElementById('estado');
const cameraRef = { current: null };

// -------------------- SISTEMA DE NOTIFICACIONES (TOAST) --------------------
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    
    // Crear elemento
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '❌';

    toast.innerHTML = `<span class="toast-icon">${icon}</span> <span>${message}</span>`;
    
    container.appendChild(toast);

    // Eliminar del DOM después de la animación (3s)
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// -------------------- SESIÓN USUARIO --------------------
async function checkUserSession() {
    const { data: { session }, error } = await supabase.auth.getSession();
    if (error || !session || !session.user) {
        window.location.href = 'index.html';
        return;
    }
    const user = session.user;
    const userEmail = document.getElementById('userEmail');
    if (userEmail) userEmail.value = user.email;
    
    // Al cargar sesión, revisamos la salud histórica
    checkMedicalHealth(user.id);
}

checkUserSession();

async function getUserRole() {
    try {
        const { data: { user } } = await supabase.auth.getUser();
        if (!user) return 'User';
        const { data } = await supabase.from('Usuarios').select('rol').eq('id_usuario', user.id).single();
        return data ? data.rol : 'User';
    } catch { return 'User'; }
}

// Variable global de seguridad
let isRecoveryMode = false;

supabase.auth.onAuthStateChange(async (event, session) => {
    if (event === 'SIGNED_OUT') {
        window.location.href = 'index.html';
    } 
    
    // DETECCIÓN SEGURA DE RECUPERACIÓN
    if (event === 'PASSWORD_RECOVERY') {
        isRecoveryMode = true;
        
        // 1. Mostrar mensaje al usuario
        showToast('Modo Recuperación: Crea tu nueva contraseña', 'info');
        
        // 2. Llevarlo directamente a la sección de perfil
        const usuariosBtn = document.querySelector('.menu-btn[data-target="usuarios"]');
        if(usuariosBtn) usuariosBtn.click();
        
        // 3. Enfocar el campo de nueva contraseña
        setTimeout(() => {
            document.getElementById('newPassword').focus();
            // Opcional: Ocultar el campo de "Contraseña actual" visualmente para no confundir
            document.getElementById('currentPassword').parentElement.style.display = 'none';
        }, 500);
    }
});

document.getElementById('logoutBtn').addEventListener('click', async () => {
    await supabase.auth.signOut();
    window.location.href = 'index.html';
});

// -------------------- LÓGICA DE SALUD Y RECOMENDACIONES (15 DÍAS) --------------------
async function checkMedicalHealth(userId) {
    const card = document.getElementById('medicalAlertCard');
    
    // 1. ¿Existe recomendación activa?
    const { data: lastRec } = await supabase
        .from('recomendaciones_medicas')
        .select('*')
        .eq('id_usuario', userId)
        .order('fecha_generacion', { ascending: false })
        .limit(1)
        .single();

    if (lastRec) {
        const dias = (new Date() - new Date(lastRec.fecha_generacion)) / (1000 * 60 * 60 * 24);
        
        if (lastRec.estado === 'Atendida' && dias < 30) return; // Periodo de gracia
        if (lastRec.estado === 'Omitida' && dias < 3) return; // No insistir pronto
        
        if (lastRec.estado === 'Pendiente') {
            showMedicalCard(lastRec.id_recomendacion, lastRec.descripcion);
            return;
        }
    }

    // 2. Análisis de 15 días
    const fifteenDaysAgo = new Date();
    fifteenDaysAgo.setDate(fifteenDaysAgo.getDate() - 15);

    const { data: sessions } = await supabase
        .from('sesiones_conduccion')
        .select('nivel_riesgo_final')
        .eq('id_usuario', userId)
        .gte('fecha_inicio', fifteenDaysAgo.toISOString());

    if (!sessions || sessions.length < 5) return; // Mínimo 5 viajes

    // Filtrar nulos para evitar errores matemáticos
    const validSessions = sessions.filter(s => s.nivel_riesgo_final !== null);
    
    let badSessions = 0;
    validSessions.forEach(s => {
        // Normalización por si acaso
        const riesgo = s.nivel_riesgo_final;
        if (riesgo === 'Alto riesgo' || riesgo === 'Alto' || riesgo === 'Moderado') {
            badSessions++;
        }
    });

    const fatiguePercentage = (badSessions / validSessions.length) * 100;

    // UMBRAL: 40%
    if (fatiguePercentage >= 40) {
        const desc = `Hola, hemos notado que en las últimas dos semanas, el ${fatiguePercentage.toFixed(0)}% de tus viajes presentaron indicadores de cansancio frecuente.`;
        
        const { data: newRec } = await supabase
            .from('recomendaciones_medicas')
            .insert([{
                id_usuario: userId,
                motivo: 'Fatiga Recurrente',
                descripcion: desc,
                estado: 'Pendiente',
                rango_analizado: '15 dias'
            }])
            .select().single();

        if (newRec) showMedicalCard(newRec.id_recomendacion, desc);
    }
}

function showMedicalCard(recId, description) {
    const card = document.getElementById('medicalAlertCard');
    const text = document.getElementById('medText');
    
    text.textContent = description + " Te sugerimos visitar a un especialista para descartar condiciones como astenia y asegurarnos de que estés al 100%.";
    card.style.display = 'flex';

    document.getElementById('btnMedYes').onclick = async () => {
        await supabase.from('recomendaciones_medicas').update({ estado: 'Atendida' }).eq('id_recomendacion', recId);
        card.style.display = 'none';
        showToast("¡Excelente! Nos alegra saber que te cuidas.", "success"); // <--- TOAST
    };

    document.getElementById('btnMedNo').onclick = async () => {
        await supabase.from('recomendaciones_medicas').update({ estado: 'Omitida' }).eq('id_recomendacion', recId);
        card.style.display = 'none';
        showToast("Entendido. Te lo recordaremos más adelante.", "info"); // <--- TOAST
    };
}

// -------------------- SESIÓN DE CONDUCCIÓN --------------------
async function startUserSession() {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return;

    const { data, error } = await supabase
        .from('sesiones_conduccion')
        .insert([{ id_usuario: user.id, fecha_inicio: new Date().toISOString() }])
        .select().single();

    if (!error) {
        sessionId = data.id_sesion;
        console.log('Sesión iniciada:', sessionId);
        showToast("Sesión iniciada correctamente", "success");
    } else {
        showToast("Error al conectar con la base de datos", "error");
    }
}

async function endUserSession() {
    if (!sessionId) return;

    let riesgoFinal = 'Normal';
    if (maxSessionRisk === 3) riesgoFinal = 'Alto riesgo';
    else if (maxSessionRisk === 2) riesgoFinal = 'Moderado';
    else if (maxSessionRisk === 1) riesgoFinal = 'Leve';

    await supabase
        .from('sesiones_conduccion')
        .update({ 
            fecha_fin: new Date().toISOString(),
            nivel_riesgo_final: riesgoFinal
        })
        .eq('id_sesion', sessionId);

    sessionId = null;
}

// -------------------- BOTONES DETECCIÓN --------------------
document.getElementById('startDetection').addEventListener('click', async () => {
    const rol = await getUserRole();
    if (rol === 'Dev') canvasElement.style.display = 'block';
    
    maxSessionRisk = 0;

    await startUserSession(); 
    if (!sessionId) return; // El toast de error ya salió en startUserSession

    videoElement.style.display = 'block';
    
    startDetection({ 
        rol, videoElement, canvasElement, estado, cameraRef, sessionId,
        onRiskUpdate: (level) => {
            let val = 0;
            if (level === 'Leve') val = 1;
            if (level === 'Moderado') val = 2;
            if (level === 'Alto riesgo') val = 3;
            if (val > maxSessionRisk) maxSessionRisk = val;
        }
    });

    document.getElementById('startDetection').style.display = 'none';
    document.getElementById('stopDetection').style.display = 'inline-block';
});

document.getElementById('stopDetection').addEventListener('click', async () => {
    stopDetection(cameraRef);
    videoElement.style.display = 'none';
    canvasElement.style.display = 'none';

    await endUserSession();

    estado.innerHTML = "<p>Detección detenida.</p>";
    document.getElementById('startDetection').style.display = 'inline-block';
    document.getElementById('stopDetection').style.display = 'none';

    showPostSessionModal();
});

function showPostSessionModal() {
    const modal = document.getElementById('recommendationModal');
    const icon = document.getElementById('recIcon');
    const title = document.getElementById('recSubtitle');
    const text = document.getElementById('recText');

    if (maxSessionRisk === 3) {
        icon.textContent = "🛑";
        title.textContent = "Cuidado";
        text.textContent = "¡Cuidado! Hubo momentos donde el sueño casi te gana. Por tu seguridad, es mejor llegar tarde que no llegar. Tómate un descanso.";
    } else if (maxSessionRisk === 2) {
        icon.textContent = "⚠️";
        title.textContent = "Atención";
        text.textContent = "Oye, notamos que te dio algo de sueño. Los bostezos nos delatan. Quizás sea momento de una pausa o un café.";
    } else if (maxSessionRisk === 1) {
        icon.textContent = "💤";
        title.textContent = "Un poco de cansancio";
        text.textContent = "Parece que hoy el viaje estuvo un poco pesado. Notamos cansancio en tus ojos. Intenta dormir mejor hoy.";
    } else {
        icon.textContent = "🚗";
        title.textContent = "¡Excelente viaje!";
        text.textContent = "Todo marcha sobre ruedas. Has conducido con muy buena atención.";
    }

    modal.classList.add('active');
}

// Cerrar modales
document.getElementById('closeRecModal').onclick = () => document.getElementById('recommendationModal').classList.remove('active');
document.getElementById('btnCloseRec').onclick = () => document.getElementById('recommendationModal').classList.remove('active');

// -------------------- PERFIL DE USUARIO --------------------
// (Tu código de perfil se mantiene igual, solo asegúrate de cambiar los alert por showToast si hay alguno)
async function loadUserProfile() {
    const { data: authData } = await supabase.auth.getUser();
    if (!authData.user) return;
    const { data } = await supabase.from('Usuarios').select('nombre').eq('id_usuario', authData.user.id).single();
    if (data) document.getElementById('userName').value = data.nombre;
    document.getElementById('userEmail').value = authData.user.email;
}
document.querySelector('.menu-btn[data-target="usuarios"]').addEventListener('click', loadUserProfile);

document.getElementById('editProfileForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const messageEl = document.getElementById('profileMessage');
    if(messageEl) messageEl.textContent = ''; 

    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return;

    const newName = document.getElementById('userName').value.trim();
    const newEmail = document.getElementById('userEmail').value.trim();
    const newPassword = document.getElementById('newPassword').value;
    const repeatPassword = document.getElementById('repeatPassword').value;
    const currentPassword = document.getElementById('currentPassword').value;

    try {
        // --- VALIDACIÓN DE SEGURIDAD ---
        
        // Caso A: Usuario Normal (Debe poner su contraseña actual para cualquier cambio sensible)
        if (!isRecoveryMode) {
            if (!currentPassword) {
                throw new Error('Por seguridad, ingresa tu contraseña actual para guardar cambios.');
            }
            
            // Verificar que la contraseña actual sea correcta
            const { error: authError } = await supabase.auth.signInWithPassword({
                email: user.email,
                password: currentPassword
            });
            if (authError) throw new Error('La contraseña actual es incorrecta');
        }
        
        // Caso B: Modo Recuperación (Viene del correo, no pedimos la anterior)
        // (El código pasa directo sin entrar al if de arriba)

        // ----------------------------------------

        // 1. Actualizar Nombre
        const { data: existingUser } = await supabase
            .from('Usuarios').select('id_usuario').eq('id_usuario', user.id).single();

        if (!existingUser) {
            await supabase.from('Usuarios').insert([{ id_usuario: user.id, nombre: newName }]);
        } else {
            await supabase.from('Usuarios').update({ nombre: newName }).eq('id_usuario', user.id);
        }

        // 2. Actualizar Email
        if (newEmail && newEmail !== user.email) {
            const { error } = await supabase.auth.updateUser({ email: newEmail });
            if (error) throw error;
        }

        // 3. Actualizar Contraseña
        if (newPassword || repeatPassword) {
            if (newPassword.length < 6) throw new Error('La nueva contraseña debe tener al menos 6 caracteres');
            if (newPassword !== repeatPassword) throw new Error('Las nuevas contraseñas no coinciden');

            const { error } = await supabase.auth.updateUser({ password: newPassword });
            if (error) throw error;
            
            showToast('¡Contraseña actualizada con éxito!', 'success');
            
            // Si estábamos en modo recuperación, lo apagamos y restauramos la UI
            if (isRecoveryMode) {
                isRecoveryMode = false;
                document.getElementById('currentPassword').parentElement.style.display = 'flex';
                showToast('Tu cuenta ya es segura. Próximos cambios requerirán tu contraseña.', 'info');
            }
        } else {
            showToast('Perfil actualizado correctamente', 'success');
        }

        // Limpiar campos
        document.getElementById('newPassword').value = '';
        document.getElementById('repeatPassword').value = '';
        document.getElementById('currentPassword').value = '';

    } catch (err) {
        console.error(err);
        showToast(err.message || 'Error al actualizar perfil', 'error');
    }
});

// -------------------- GENERACIÓN DE REPORTES PDF --------------------
// Nota: jspdf se cargó desde el CDN en el HTML, así que usamos window.jspdf

document.getElementById('btnDownloadPDF').addEventListener('click', generatePDFReport);

// --- Función auxiliar para cargar la imagen del logo ---
    function loadImage(url) {
        return new Promise((resolve) => {
            const img = new Image();
            img.src = url;
            img.crossOrigin = "Anonymous"; 
            img.onload = () => resolve(img);
            img.onerror = () => resolve(null); // Si falla, sigue sin logo
        });
    }

    async function generatePDFReport() {
        const startDate = document.getElementById('reportStartDate').value;
        const endDate = document.getElementById('reportEndDate').value;

        if (!startDate || !endDate) {
            showToast("Selecciona un rango de fechas para el informe", "error");
            return;
        }

        showToast("Generando informe oficial...", "info");

        // 1. Obtener Usuario (Auth) y Perfil (Nombre)
        const { data: { user } } = await supabase.auth.getUser();
        if (!user) return;

        // Consultamos el nombre en la tabla Usuarios
        const { data: perfil } = await supabase
            .from('Usuarios')
            .select('nombre')
            .eq('id_usuario', user.id)
            .single();
        
        const nombreUsuario = perfil ? perfil.nombre : 'Conductor';
        const correoUsuario = user.email;

        // --- Cargar Logo (Esperamos a que cargue) ---
        // Usamos el logo blanco porque el fondo del header será oscuro
        const logoImg = await loadImage('img/white-logo.png');

        // --- 2. Rango de Fechas Exacto ---
        const startISO = new Date(startDate + 'T00:00:00').toISOString();
        const endISO = new Date(endDate + 'T23:59:59.999').toISOString();

        // --- 3. Obtener Datos (Sesiones) ---
        const { data: sesiones, error: sessError } = await supabase
            .from('sesiones_conduccion')
            .select('*')
            .eq('id_usuario', user.id)
            .gte('fecha_inicio', startISO)
            .lte('fecha_inicio', endISO)
            .order('fecha_inicio', { ascending: true });

        if (sessError || !sesiones || sesiones.length === 0) {
            showToast("No hay registros en esas fechas", "error");
            return;
        }

        const sessionIds = sesiones.map(s => s.id_sesion);
        
        // Alertas
        const { data: alertas } = await supabase
            .from('Alertas')
            .select('*')
            .in('id_sesion', sessionIds)
            .order('fecha_alerta', { ascending: true });

        // Recomendaciones
        const { data: recomendaciones } = await supabase
            .from('recomendaciones_medicas')
            .select('*')
            .eq('id_usuario', user.id)
            .gte('fecha_generacion', startISO)
            .lte('fecha_generacion', endISO);

        // --- 4. Cálculos Estadísticos ---
        let totalMinutos = 0;
        let sesionesAltoRiesgo = 0;

        sesiones.forEach(s => {
            if (s.duracion_min) totalMinutos += s.duracion_min;
            const estado = s.nivel_riesgo_final || '';
            if (estado.includes('Alto')) sesionesAltoRiesgo++;
        });

        // --- 5. Generación del PDF ---
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        
        // Colores
        const darkBlue = [15, 23, 42]; 
        const brightBlue = [59, 130, 246];

        // --- HEADER CON LOGO Y DATOS ---
        doc.setFillColor(...darkBlue);
        doc.rect(0, 0, 210, 50, 'F'); // Aumenté un poco el alto a 50
        
        // 1. Logo (Si cargó)
        if (logoImg) {
            // x=14, y=10, width=25, height=auto (según ratio)
            const imgRatio = logoImg.height / logoImg.width;
            doc.addImage(logoImg, 'PNG', 14, 10, 30, 30 * imgRatio);
        }

        // 2. Título del Reporte (Centrado un poco a la derecha del logo)
        doc.setTextColor(255, 255, 255);
        doc.setFontSize(20);
        doc.text("INFORME DE ESTADO", 100, 18);
        
        doc.setFontSize(10);
        doc.setTextColor(200, 200, 200);
        doc.text(`Generado: ${new Date().toLocaleDateString()}`, 100, 24);

        // 3. Datos del Usuario (Alineados a la derecha del header)
        doc.setFontSize(11);
        doc.setTextColor(255, 255, 255);
        doc.text(nombreUsuario.toUpperCase(), 190, 38, { align: 'right' }); // Nombre
        
        doc.setFontSize(9);
        doc.setTextColor(156, 163, 175); // Gris claro
        doc.text(correoUsuario, 190, 43, { align: 'right' }); // Correo

        // --- TABLA RESUMEN ---
        doc.setTextColor(0, 0, 0);
        doc.autoTable({
            startY: 60,
            head: [['Métrica', 'Resultado']],
            body: [
                ['Periodo Analizado', `${startDate} al ${endDate}`],
                ['Total Sesiones Realizadas', sesiones.length],
                ['Tiempo Total Monitoreado', `${totalMinutos} minutos`],
                ['Sesiones Críticas (Alto Riesgo)', sesionesAltoRiesgo], 
                ['Total Eventos de Alerta', alertas ? alertas.length : 0]
            ],
            theme: 'grid',
            headStyles: { fillColor: brightBlue, halign: 'center' },
            columnStyles: { 0: { fontStyle: 'bold' }, 1: { halign: 'center' } }
        });

        // --- BITÁCORA DE ALERTAS ---
        let finalY = doc.lastAutoTable.finalY + 15;
        doc.setFontSize(14);
        doc.setTextColor(...brightBlue);
        doc.text("Bitácora Detallada de Eventos", 14, finalY);

        if (alertas && alertas.length > 0) {
            const bodyAlertas = alertas.map(a => {
                let valorFormateado = '-';
                if (a.valor_medido !== null) {
                    const val = parseFloat(a.valor_medido).toFixed(1);
                    if (a.causa_detonante.includes('Microsueño')) valorFormateado = `${val} seg`;
                    else if (a.causa_detonante.includes('Fatiga')) valorFormateado = `${val} parp/min`;
                    else if (a.causa_detonante.includes('Bostezos')) valorFormateado = `${val} detectados`;
                    else if (a.causa_detonante.includes('Somnolencia')) valorFormateado = `${val} eventos`;
                    else valorFormateado = val; 
                }

                return [
                    new Date(a.fecha_alerta).toLocaleDateString() + ' ' + new Date(a.fecha_alerta).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}),
                    a.causa_detonante || 'General',
                    a.nivel_riesgo,
                    valorFormateado
                ];
            });

            doc.autoTable({
                startY: finalY + 5,
                head: [['Fecha / Hora', 'Evento', 'Gravedad', 'Medición']],
                body: bodyAlertas,
                theme: 'striped',
                headStyles: { fillColor: [220, 38, 38] },
                didParseCell: function(data) {
                    if (data.section === 'body' && data.column.index === 2) {
                        if (data.cell.raw === 'Alto riesgo') data.cell.styles.textColor = [220, 38, 38];
                        else if (data.cell.raw === 'Moderado') data.cell.styles.textColor = [217, 119, 6];
                    }
                }
            });
        } else {
            doc.setFontSize(11);
            doc.setTextColor(100);
            doc.text("No se registraron alertas en este periodo.", 14, finalY + 10);
        }

        // --- DETALLE DE SESIONES ---
        finalY = doc.lastAutoTable.finalY + 15;
        if (finalY > 250) { doc.addPage(); finalY = 20; }

        doc.setFontSize(14);
        doc.setTextColor(...brightBlue);
        doc.text("Historial de Sesiones", 14, finalY);

        const bodySesiones = sesiones.map(s => {
            let textoDuracion = 'En curso';

            // 1. Verificamos si la sesión ya terminó (tiene fecha_fin)
            if (s.fecha_fin) {
                // Si la base de datos calculó 1 o más minutos
                if (s.duracion_min && s.duracion_min > 0) {
                    textoDuracion = s.duracion_min + ' min';
                } else {
                    // Si terminó pero la BD dice 0 (fueron segundos)
                    textoDuracion = '< 1 min';
                }
            }

            return [
                new Date(s.fecha_inicio).toLocaleDateString() + ' ' + new Date(s.fecha_inicio).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}),
                textoDuracion,
                s.nivel_riesgo_final || 'Incompleta' // Si terminó pero no tiene riesgo, ponemos Incompleta
            ];
        });

        doc.autoTable({
            startY: finalY + 5,
            head: [['Inicio', 'Duración', 'Estado Final']],
            body: bodySesiones,
            theme: 'striped',
            headStyles: { fillColor: [71, 85, 105] },
            didParseCell: function(data) {
                if (data.section === 'body' && data.column.index === 2) {
                    if (data.cell.raw && data.cell.raw.includes('Alto')) {
                        data.cell.styles.textColor = [220, 38, 38];
                        data.cell.styles.fontStyle = 'bold';
                    }
                }
            }
        });

        // --- RECOMENDACIONES (Si existen) ---
        if (recomendaciones && recomendaciones.length > 0) {
            doc.addPage();
            doc.setFontSize(14);
            doc.setTextColor(...brightBlue);
            doc.text("Historial de Recomendaciones Médicas", 14, 20);

            const bodyRecs = recomendaciones.map(r => [
                new Date(r.fecha_generacion).toLocaleDateString(),
                r.motivo,
                r.estado,
                r.descripcion
            ]);

            doc.autoTable({
                startY: 25,
                head: [['Fecha', 'Motivo', 'Estado', 'Descripción']],
                body: bodyRecs,
                theme: 'grid',
                headStyles: { fillColor: [234, 179, 8], textColor: [0,0,0] }, // Amarillo
                columnStyles: { 3: { cellWidth: 80 } }
            });
        }

        // --- FOOTER ---
        const pageCount = doc.internal.getNumberOfPages();
        for(let i = 1; i <= pageCount; i++) {
            doc.setPage(i);
            doc.setFontSize(8);
            doc.setTextColor(150);
            doc.text(`Página ${i} de ${pageCount} - Reporte VisioGuard - ${nombreUsuario}`, 105, 290, { align: "center" });
        }

        doc.save(`VisioGuard_${nombreUsuario.replace(/\s+/g, '_')}_${startDate}.pdf`);
        showToast("Informe descargado correctamente", "success");
    }

// -------------------- MODO NOCTURNO (POTENCIAR CÁMARA) --------------------
document.getElementById('nightModeToggle').addEventListener('change', (e) => {
    const isEnabled = e.target.checked;

    // 1. Activar lógica interna (Para que la IA detecte mejor)
    toggleNightMode(isEnabled);

    // 2. Activar efecto visual (Para que TÚ veas la diferencia)
    const videoEl = document.querySelector('.input_video');
    
    if (isEnabled) {
        videoEl.classList.add('night-vision-effect');
        showToast("Modo Nocturno Activado: +Brillo +Contraste", "info");
    } else {
        videoEl.classList.remove('night-vision-effect');
        showToast("Modo Nocturno Desactivado", "info");
    }
});