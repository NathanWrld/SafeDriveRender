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
    // ... (tu lógica de actualización) ...
    // Al final, en lugar de messageEl.textContent, puedes usar:
    // showToast('Perfil actualizado correctamente', 'success');
});