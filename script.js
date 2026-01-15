import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm'
import { startDetection, stopDetection } from './detection.js';

const supabaseUrl = 'https://roogjmgxghbuiogpcswy.supabase.co'
const supabaseKey = 'sb_publishable_RTN2PXvdWOQFfUySAaTa_g_LLe-T_NU'
const supabase = createClient(supabaseUrl, supabaseKey)

// -------------------- VARIABLES GLOBALES --------------------
let sessionId = null;
const videoElement = document.querySelector('.input_video');
const canvasElement = document.querySelector('.output_canvas');
const estado = document.getElementById('estado');
const cameraRef = { current: null }; // Referencia compartida para detection.js

// -------------------- SESIÓN USUARIO --------------------
async function checkUserSession() {
    const { data: { session }, error } = await supabase.auth.getSession();
    if (error) { console.error('Error obteniendo sesión:', error.message); return; }

    if (!session || !session.user) {
        window.location.href = 'index.html';
        return;
    }

    const user = session.user;
    const userEmail = document.getElementById('userEmail');
    if (userEmail) userEmail.value = user.email;
}

checkUserSession();

async function getUserRole() {
    try {
        const { data: { user }, error: userError } = await supabase.auth.getUser();
        if (userError || !user) return 'User';

        const { data, error } = await supabase
            .from('Usuarios')
            .select('rol')
            .eq('id_usuario', user.id)
            .single();

        if (error) { console.error('Error fetch rol:', error); return 'User'; }
        console.log('Rol real de la DB:', data.rol);
        return data.rol;
    } catch (err) {
        console.error('Error obteniendo rol:', err);
        return 'User';
    }
}

supabase.auth.onAuthStateChange((event, session) => {
    if (event === 'SIGNED_OUT') window.location.href = 'index.html';
});

document.getElementById('logoutBtn').addEventListener('click', async () => {
    const { error } = await supabase.auth.signOut();
    if (error) console.error('Error cerrando sesión:', error.message);
    else window.location.href = 'index.html';
});

// -------------------- SESIÓN DE CONDUCCIÓN --------------------
async function startUserSession() {
    try {
        const { data: { user }, error: userError } = await supabase.auth.getUser();
        if (userError || !user) {
            console.error('No se pudo obtener usuario:', userError);
            alert('No se pudo obtener el usuario. Inicia sesión de nuevo.');
            return;
        }

        const { data, error } = await supabase
            .from('sesiones_conduccion')
            .insert([{ id_usuario: user.id, fecha_inicio: new Date().toISOString() }])
            .select()
            .single();

        if (error) { console.error('Error insertando sesión:', error); return; }

        sessionId = data.id_sesion;
        console.log('Sesión iniciada:', sessionId);
    } catch (error) { console.error('Error en startUserSession:', error); }
}

async function endUserSession() {
    if (!sessionId) return;

    try {
        const { error } = await supabase
            .from('sesiones_conduccion')
            .update({ fecha_fin: new Date().toISOString() })
            .eq('id_sesion', sessionId);

        if (error) console.error('Error al finalizar sesión:', error);
        else console.log('Sesión finalizada:', sessionId);

        sessionId = null;
    } catch (error) { console.error('Error en endUserSession:', error); }
}

// -------------------- BOTONES DETECCIÓN --------------------
document.getElementById('startDetection').addEventListener('click', async () => {
    const rol = await getUserRole();
    console.log('Rol detectado:', rol);

    if (rol === 'Dev') canvasElement.style.display = 'block';
    else canvasElement.style.display = 'none';
    videoElement.style.display = 'block';

    await startUserSession();
    startDetection({ rol, videoElement, canvasElement, estado, cameraRef });

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
});

// -------------------- PERFIL DE USUARIO --------------------
async function loadUserProfile() {
    const { data: authData, error: authError } = await supabase.auth.getUser();
    if (authError || !authData.user) return;

    const userId = authData.user.id;
    const { data: userProfile, error } = await supabase
        .from('Usuarios')
        .select('nombre')
        .eq('id_usuario', userId)
        .single();

    if (error) { console.error("Error cargando perfil:", error); return; }

    document.getElementById('userName').value = userProfile.nombre;
    document.getElementById('userEmail').value = authData.user.email;
}

document.querySelector('.menu-btn[data-target="usuarios"]').addEventListener('click', loadUserProfile);

document.getElementById('editProfileForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const messageEl = document.getElementById('profileMessage');
    messageEl.textContent = '';
    messageEl.style.color = '#f87171';

    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return;

    const newName = document.getElementById('userName').value.trim();
    const newEmail = document.getElementById('userEmail').value.trim();
    const newPassword = document.getElementById('newPassword').value;
    const repeatPassword = document.getElementById('repeatPassword').value;
    const currentPassword = document.getElementById('currentPassword').value;

    try {
        if (!currentPassword) throw new Error('Debes ingresar tu contraseña actual');

        const { error: authError } = await supabase.auth.signInWithPassword({
            email: user.email,
            password: currentPassword
        });

        if (authError) throw new Error('La contraseña actual es incorrecta');

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

        if (newEmail && newEmail !== user.email) {
            const { error } = await supabase.auth.updateUser({ email: newEmail });
            if (error) throw error;
        }

        if (newPassword || repeatPassword) {
            if (newPassword.length < 6) throw new Error('La contraseña debe tener al menos 6 caracteres');
            if (newPassword !== repeatPassword) throw new Error('Las contraseñas no coinciden');

            const { error } = await supabase.auth.updateUser({ password: newPassword });
            if (error) throw error;
        }

        messageEl.style.color = '#10b981';
        messageEl.textContent = 'Perfil actualizado correctamente';

        document.getElementById('newPassword').value = '';
        document.getElementById('repeatPassword').value = '';
        document.getElementById('currentPassword').value = '';

    } catch (err) {
        console.error(err);
        messageEl.textContent = err.message || 'Error al actualizar perfil';
    }
});
