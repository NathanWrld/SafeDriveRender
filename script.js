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
// --- LÓGICA DE SALUD Y RECOMENDACIONES (MODO DEBUG) ---
async function checkMedicalHealth(userId) {
    console.log("🔍 [DEBUG] Iniciando chequeo médico para usuario:", userId);

    const card = document.getElementById('medicalAlertCard');
    
    // 1. ¿Existe recomendación activa?
    const { data: lastRec, error: recError } = await supabase
        .from('recomendaciones_medicas')
        .select('*')
        .eq('id_usuario', userId)
        .order('fecha_generacion', { ascending: false })
        .limit(1)
        .single();

    if (recError && recError.code !== 'PGRST116') {
        console.error("❌ [DEBUG] Error buscando recomendaciones:", recError.message);
    }

    if (lastRec) {
        console.log("📋 [DEBUG] Encontrada recomendación previa:", lastRec);
        const dias = (new Date() - new Date(lastRec.fecha_generacion)) / (1000 * 60 * 60 * 24);
        
        if (lastRec.estado === 'Atendida' && dias < 30) {
            console.log("🛑 [DEBUG] Oculto: Ya fue atendida hace menos de 30 días.");
            return;
        }
        if (lastRec.estado === 'Omitida' && dias < 3) {
            console.log("🛑 [DEBUG] Oculto: Fue omitida hace poco.");
            return;
        }
        if (lastRec.estado === 'Pendiente') {
            console.log("✅ [DEBUG] Mostrando recomendación pendiente.");
            showMedicalCard(lastRec.id_recomendacion, lastRec.descripcion);
            return;
        }
    } else {
        console.log("ℹ️ [DEBUG] No hay recomendaciones previas.");
    }

    // 2. Análisis de 15 días
    const fifteenDaysAgo = new Date();
    fifteenDaysAgo.setDate(fifteenDaysAgo.getDate() - 15);
    console.log("📅 [DEBUG] Analizando sesiones desde:", fifteenDaysAgo.toISOString());

    const { data: sessions, error: sessError } = await supabase
        .from('sesiones_conduccion')
        .select('nivel_riesgo_final, fecha_inicio')
        .eq('id_usuario', userId)
        .gte('fecha_inicio', fifteenDaysAgo.toISOString());

    if (sessError) {
        console.error("❌ [DEBUG] Error trayendo sesiones:", sessError.message);
        return;
    }

    console.log(`🚗 [DEBUG] Sesiones encontradas: ${sessions ? sessions.length : 0}`);
    console.table(sessions); // Muestra los datos en tabla bonita en consola

    if (!sessions || sessions.length < 5) {
        console.warn("⚠️ [DEBUG] No hay suficientes sesiones (< 5) para generar alerta.");
        return; 
    }

    let badSessions = 0;
    sessions.forEach(s => {
        // Normalizamos a minúsculas por si acaso
        const riesgo = s.nivel_riesgo_final; 
        console.log(`🔎 [DEBUG] Revisando sesión: ${riesgo}`);
        
        if (riesgo === 'Alto riesgo' || riesgo === 'Moderado') {
            badSessions++;
        }
    });

    const fatiguePercentage = (badSessions / sessions.length) * 100;
    console.log(`📊 [DEBUG] Resultado: ${badSessions} malas de ${sessions.length}. Porcentaje: ${fatiguePercentage}%`);

    // UMBRAL: 40%
    if (fatiguePercentage >= 40) {
        console.log("🚨 [DEBUG] ¡UMBRAL SUPERADO! Generando alerta...");
        const desc = `Hola, hemos notado que en las últimas dos semanas, el ${fatiguePercentage.toFixed(0)}% de tus viajes presentaron indicadores de cansancio frecuente.`;
        
        const { data: newRec, error: insertError } = await supabase
            .from('recomendaciones_medicas')
            .insert([{
                id_usuario: userId,
                motivo: 'Fatiga Recurrente',
                descripcion: desc,
                estado: 'Pendiente',
                rango_analizado: '15 dias'
            }])
            .select().single();

        if (insertError) {
            console.error("❌ [DEBUG] Error creando recomendación:", insertError.message);
        } else {
            console.log("✅ [DEBUG] Recomendación creada con ID:", newRec.id_recomendacion);
            showMedicalCard(newRec.id_recomendacion, desc);
        }
    } else {
        console.log("✅ [DEBUG] El porcentaje es seguro (< 40%). No se genera alerta.");
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
        alert("¡Excelente! Nos alegra saber que te cuidas."); 
    };

    document.getElementById('btnMedNo').onclick = async () => {
        await supabase.from('recomendaciones_medicas').update({ estado: 'Omitida' }).eq('id_recomendacion', recId);
        card.style.display = 'none';
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
    }
}

async function endUserSession() {
    if (!sessionId) return;

    // Calcular riesgo final basado en lo peor que pasó en el viaje
    let riesgoFinal = 'Normal';
    if (maxSessionRisk === 3) riesgoFinal = 'Alto riesgo';
    else if (maxSessionRisk === 2) riesgoFinal = 'Moderado';
    else if (maxSessionRisk === 1) riesgoFinal = 'Leve';

    await supabase
        .from('sesiones_conduccion')
        .update({ 
            fecha_fin: new Date().toISOString(),
            nivel_riesgo_final: riesgoFinal // <--- Guardamos para el análisis médico
        })
        .eq('id_sesion', sessionId);

    sessionId = null;
    // No reseteamos maxSessionRisk aquí, lo necesitamos para el modal, se resetea al iniciar
}

// -------------------- BOTONES DETECCIÓN --------------------
document.getElementById('startDetection').addEventListener('click', async () => {
    const rol = await getUserRole();
    if (rol === 'Dev') canvasElement.style.display = 'block';
    
    // RESETEAR RIESGO AL INICIAR
    maxSessionRisk = 0;

    await startUserSession(); 
    if (!sessionId) { alert("Error de conexión al iniciar sesión."); return; }

    videoElement.style.display = 'block';
    
    startDetection({ 
        rol, videoElement, canvasElement, estado, cameraRef, sessionId,
        // CALLBACK: Escuchar cambios de riesgo en vivo
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

    // MOSTRAR FEEDBACK AMABLE
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
        text.textContent = "¡Cuidado! Hubo momentos donde el sueño casi te gana y tu seguridad es lo más importante. Es mejor llegar tarde que no llegar. Tómate un descanso, lo necesitas.";
    } else if (maxSessionRisk === 2) {
        icon.textContent = "⚠️";
        title.textContent = "Atención";
        text.textContent = "Oye, notamos que te dio algo de sueño en el camino. Los bostezos nos delatan. Quizás sea momento de tomarte una pausa o un café. ¡No te exijas de más!";
    } else if (maxSessionRisk === 1) {
        icon.textContent = "💤";
        title.textContent = "Un poco de cansancio";
        text.textContent = "Parece que hoy el viaje estuvo un poco pesado. Notamos cansancio en tus ojos. ¿Qué tal si hoy intentas irte a la cama un poquito antes?";
    } else {
        icon.textContent = "🚗";
        title.textContent = "¡Excelente viaje!";
        text.textContent = "Todo marcha sobre ruedas. Has conducido con muy buena atención. Sigue descansando bien para mantener ese ritmo.";
    }

    modal.classList.add('active');
}

// Cerrar modales
document.getElementById('closeRecModal').onclick = () => document.getElementById('recommendationModal').classList.remove('active');
document.getElementById('btnCloseRec').onclick = () => document.getElementById('recommendationModal').classList.remove('active');

// -------------------- PERFIL DE USUARIO (Código existente minimizado) --------------------
async function loadUserProfile() { /* ... tu código de perfil ... */ }
document.querySelector('.menu-btn[data-target="usuarios"]').addEventListener('click', loadUserProfile);
document.getElementById('editProfileForm').addEventListener('submit', async (e) => { /* ... tu código de editar ... */ });