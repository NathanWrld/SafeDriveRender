import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm'
import { startDetection, stopDetection } from './detection.js';

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

supabase.auth.onAuthStateChange((event) => {
    if (event === 'SIGNED_OUT') window.location.href = 'index.html';
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
    
    text.textContent = description + " Te sugerimos cariñosamente visitar a un especialista para descartar condiciones como astenia y asegurarnos de que estés al 100%.";
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
    
    // Limpiamos mensajes antiguos del formulario por si acaso
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
        if (!currentPassword) throw new Error('Debes ingresar tu contraseña actual');

        // 1. Verificar contraseña actual
        const { error: authError } = await supabase.auth.signInWithPassword({
            email: user.email,
            password: currentPassword
        });

        if (authError) throw new Error('La contraseña actual es incorrecta');

        // 2. Verificar/Crear registro en tabla Usuarios
        const { data: existingUser, error: fetchError } = await supabase
            .from('Usuarios')
            .select('id_usuario')
            .eq('id_usuario', user.id)
            .single();

        if (fetchError && fetchError.code !== 'PGRST116') throw fetchError;

        if (!existingUser) {
            const { error } = await supabase
                .from('Usuarios')
                .insert([{ id_usuario: user.id, nombre: newName }]);
            if (error) throw error;
        } else {
            const { error } = await supabase
                .from('Usuarios')
                .update({ nombre: newName })
                .eq('id_usuario', user.id);
            if (error) throw error;
        }

        // 3. Actualizar Email (si cambió)
        if (newEmail && newEmail !== user.email) {
            const { error } = await supabase.auth.updateUser({ email: newEmail });
            if (error) throw error;
        }

        // 4. Actualizar Password (si cambió)
        if (newPassword || repeatPassword) {
            if (newPassword.length < 6) throw new Error('La contraseña debe tener al menos 6 caracteres');
            if (newPassword !== repeatPassword) throw new Error('Las contraseñas no coinciden');

            const { error } = await supabase.auth.updateUser({ password: newPassword });
            if (error) throw error;
        }

        // --- ÉXITO CON TOAST ---
        showToast('Perfil actualizado correctamente', 'success');

        // Limpiar campos de contraseña
        document.getElementById('newPassword').value = '';
        document.getElementById('repeatPassword').value = '';
        document.getElementById('currentPassword').value = '';

    } catch (err) {
        console.error(err);
        // --- ERROR CON TOAST ---
        showToast(err.message || 'Error al actualizar perfil', 'error');
    }
});

// -------------------- GENERACIÓN DE REPORTES PDF --------------------
// Nota: jspdf se cargó desde el CDN en el HTML, así que usamos window.jspdf

document.getElementById('btnDownloadPDF').addEventListener('click', generatePDFReport);

async function generatePDFReport() {
        const startDate = document.getElementById('reportStartDate').value;
        const endDate = document.getElementById('reportEndDate').value;

        if (!startDate || !endDate) {
            showToast("Selecciona un rango de fechas para el informe", "error");
            return;
        }

        showToast("Generando informe, por favor espera...", "info");

        const { data: { user } } = await supabase.auth.getUser();
        if (!user) return;

        // --- CORRECCIÓN DE FECHAS (FIX ZONA HORARIA) ---
        // Forzamos el inicio al primer segundo y el fin al último segundo del día local
        const startISO = new Date(startDate + 'T00:00:00').toISOString();
        const endISO = new Date(endDate + 'T23:59:59.999').toISOString();

        console.log("📅 Generando reporte desde:", startISO, "hasta:", endISO);

        // 1. Obtener Datos: Sesiones
        const { data: sesiones, error: sessError } = await supabase
            .from('sesiones_conduccion')
            .select('*')
            .eq('id_usuario', user.id)
            .gte('fecha_inicio', startISO) // Usamos las fechas corregidas
            .lte('fecha_inicio', endISO)
            .order('fecha_inicio', { ascending: true });

        if (sessError || !sesiones || sesiones.length === 0) {
            showToast("No hay datos en el rango seleccionado", "error");
            console.log("Error o vacío:", sessError);
            return;
        }

        // 2. Obtener Alertas
        const sessionIds = sesiones.map(s => s.id_sesion);
        const { data: alertas } = await supabase
            .from('Alertas')
            .select('*')
            .in('id_sesion', sessionIds)
            .order('fecha_alerta', { ascending: true });

        // 3. Obtener Recomendaciones
        const { data: recomendaciones } = await supabase
            .from('recomendaciones_medicas')
            .select('*')
            .eq('id_usuario', user.id)
            .gte('fecha_generacion', startISO)
            .lte('fecha_generacion', endISO);

        // --- CÁLCULOS Y DIAGNÓSTICO ---
        let totalMinutos = 0;
        let sesionesAltoRiesgo = 0;

        sesiones.forEach(s => {
            if (s.duracion_min) totalMinutos += s.duracion_min;
            
            // CORRECCIÓN DE CONTEO: Limpiamos espacios y verificamos
            const riesgo = s.nivel_riesgo_final ? s.nivel_riesgo_final.trim() : '';
            console.log(`Evaluando sesión ${s.id_sesion}: Estado = "${riesgo}"`); // <--- MIRA ESTO EN CONSOLA

            if (riesgo === 'Alto riesgo' || riesgo === 'Alto') {
                sesionesAltoRiesgo++;
            }
        });

        // --- CREACIÓN DEL PDF ---
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        const colorPrimary = [15, 23, 42]; 
        const colorAccent = [59, 130, 246];

        // Header
        doc.setFillColor(...colorPrimary);
        doc.rect(0, 0, 210, 40, 'F');
        
        doc.setTextColor(255, 255, 255);
        doc.setFontSize(22);
        doc.text("INFORME DE FATIGA", 105, 15, { align: "center" });
        
        doc.setFontSize(10);
        doc.text(`Generado: ${new Date().toLocaleString()}`, 105, 25, { align: "center" });
        doc.text(`Rango: ${startDate} al ${endDate}`, 105, 32, { align: "center" });

        // Resumen
        doc.setTextColor(0, 0, 0);
        doc.autoTable({
            startY: 50,
            head: [['Métrica Clave', 'Valor']],
            body: [
                ['Total de Sesiones Analizadas', sesiones.length],
                ['Tiempo Total al Volante', `${totalMinutos} minutos`],
                ['Sesiones Críticas (Alto Riesgo)', sesionesAltoRiesgo], // <--- AHORA DEBERÍA SALIR BIEN
                ['Total de Alertas Registradas', alertas ? alertas.length : 0]
            ],
            theme: 'grid',
            headStyles: { fillColor: colorAccent, halign: 'center' },
            columnStyles: { 1: { fontStyle: 'bold', halign: 'center' } }
        });

        // Metodología
        let finalY = doc.lastAutoTable.finalY + 10;
        doc.setFontSize(9);
        doc.setTextColor(100);
        const metodoText = "NOTA TÉCNICA: 'Sesión Crítica' se define como un trayecto donde el sistema detectó microsueños (EAR < 0.20) o fatiga severa persistente que obligó a detener la medición en estado de Alto Riesgo.";
        doc.text(metodoText, 14, finalY, { maxWidth: 180, align: 'justify' });

        // Tabla Detallada
        finalY += 15;
        doc.setFontSize(14);
        doc.setTextColor(...colorAccent);
        doc.text("Desglose de Sesiones", 14, finalY);

        const bodySesiones = sesiones.map(s => [
            new Date(s.fecha_inicio).toLocaleDateString() + ' ' + new Date(s.fecha_inicio).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}),
            s.duracion_min ? s.duracion_min + ' min' : 'Running',
            s.nivel_riesgo_final || 'Incompleta'
        ]);

        doc.autoTable({
            startY: finalY + 5,
            head: [['Fecha / Hora', 'Duración', 'Estado Final']],
            body: bodySesiones,
            theme: 'striped',
            headStyles: { fillColor: [71, 85, 105] },
            didParseCell: function(data) {
                if (data.section === 'body' && data.column.index === 2) {
                    const val = data.cell.raw;
                    if (val === 'Alto riesgo' || val === 'Alto') {
                        data.cell.styles.textColor = [220, 38, 38]; // Rojo
                        data.cell.styles.fontStyle = 'bold';
                    } else if (val === 'Moderado') {
                        data.cell.styles.textColor = [234, 179, 8]; // Amarillo oscuro
                    }
                }
            }
        });

        // Alertas (si hay)
        if (alertas && alertas.length > 0) {
            doc.addPage();
            doc.setFontSize(14);
            doc.setTextColor(...colorAccent);
            doc.text("Bitácora de Eventos (Alertas)", 14, 20);
            
            const bodyAlertas = alertas.map(a => [
                new Date(a.fecha_alerta).toLocaleTimeString(),
                a.causa_detonante || 'General',
                a.nivel_riesgo,
                a.valor_medido ? a.valor_medido.toString() : '-'
            ]);

            doc.autoTable({
                startY: 25,
                head: [['Hora', 'Evento', 'Gravedad', 'Valor']],
                body: bodyAlertas,
                theme: 'grid',
                headStyles: { fillColor: [239, 68, 68] }
            });
        }

        doc.save(`Informe_VisioGuard_${startDate}_${endDate}.pdf`);
        showToast("Informe PDF descargado con éxito", "success");
    }