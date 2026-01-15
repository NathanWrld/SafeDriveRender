document.addEventListener('DOMContentLoaded', () => {
    const burgerBtn = document.getElementById('burgerBtn');
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('overlay');
    const closeSidebarBtn = document.getElementById('closeSidebar');

    // === FUNCIÓN GENERAL PARA CERRAR SLIDE ===
    function closeSidebar() {
        sidebar.classList.remove('active');
        overlay.classList.remove('active');
        closeSidebarBtn.classList.remove('active');
    }

    // === ABRIR SLIDE ===
    burgerBtn.addEventListener('click', () => {
        sidebar.classList.add('active');
        overlay.classList.add('active');
        closeSidebarBtn.classList.add('active');
    });

    // === CERRAR CON OVERLAY ===
    overlay.addEventListener('click', closeSidebar);

    // === CERRAR CON BOTÓN X ===
    closeSidebarBtn.addEventListener('click', closeSidebar);

    // === CERRAR CON ESCAPE ===
    document.addEventListener('keydown', (e) => {
        if (e.key === "Escape") closeSidebar();
    });

    // === CERRAR AL SELECCIONAR ITEM DEL MENÚ (solo móvil) ===
    document.querySelectorAll('.menu-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            if (window.innerWidth <= 768) closeSidebar();
        });
    });
});
