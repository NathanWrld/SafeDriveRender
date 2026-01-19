/* dashboard.js */
import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm'

// --- 1. CONFIGURACIÓN SUPABASE ---
const supabaseUrl = 'https://roogjmgxghbuiogpcswy.supabase.co'
const supabaseKey = 'sb_publishable_RTN2PXvdWOQFfUySAaTa_g_LLe-T_NU'
const supabase = createClient(supabaseUrl, supabaseKey)

// --- 2. VARIABLES GLOBALES ---
let myChart = null;
let currentUserId = null;

// Elementos de la UI
const tableBody = document.getElementById('historyTable');
const prevBtn = document.getElementById('prevPage');
const nextBtn = document.getElementById('nextPage');
const pageInfo = document.getElementById('pageInfo');
const btnFilter = document.getElementById('applyFilters');

// Variables de paginación
let currentPage = 1;
const pageSize = 10;

// Variables de Tema
const themeBtn = document.getElementById('themeToggle');
const headerLogo = document.getElementById('headerLogo');
const sidebarLogo = document.getElementById('sidebarLogo');


// --- 3. INICIALIZACIÓN ---
async function initDashboard() {
    // Verificar sesión
    const { data: { user } } = await supabase.auth.getUser();
    if (user) {
        currentUserId = user.id; 
        
        // Cargar tema guardado
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            document.body.setAttribute('data-theme', 'light');
            updateLogos(true);
        }

        // Cargar datos
        loadChartData(user.id);
        fetchHistory(1);
    } else {
        // Opcional: Redirigir si no hay usuario
        // window.location.href = 'index.html';
    }
}

// --- 4. EVENT LISTENERS ---

// Navegación Sidebar
const menuBtns = document.querySelectorAll('.menu-btn');
menuBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        menuBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Mostrar sección correspondiente
        ['inicio','deteccion','usuarios','historial'].forEach(sec => {
            document.getElementById(sec).style.display = (sec === btn.dataset.target) ? 'block' : 'none';
        });

        // FIX GRÁFICA: Si es inicio, redimensionar
        if(btn.dataset.target === 'inicio') {
            setTimeout(() => {
                if(myChart) {
                    myChart.resize();
                    myChart.update();
                }
            }, 50);
        }
    });
});

// Botón de Tema
if(themeBtn) {
    themeBtn.addEventListener('click', () => {
        const currentTheme = document.body.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        
        if (newTheme === 'light') {
            document.body.setAttribute('data-theme', 'light');
        } else {
            document.body.removeAttribute('data-theme');
        }

        updateLogos(newTheme === 'light');
        localStorage.setItem('theme', newTheme);
        
        // Actualizar colores de la gráfica
        if(myChart) {
            const textColor = newTheme === 'light' ? '#475569' : '#94a3b8';
            const gridColor = newTheme === 'light' ? '#e2e8f0' : '#334155';
            
            myChart.options.scales.x.ticks.color = textColor;
            myChart.options.scales.y.ticks.color = textColor;
            myChart.options.scales.x.grid.color = gridColor;
            myChart.options.scales.y.grid.color = gridColor;
            myChart.update();
        }
    });
}

function updateLogos(isLight) {
    const logoSrc = isLight ? 'img/blue-logo.png' : 'img/white-logo.png';
    if(headerLogo) headerLogo.src = logoSrc;
    if(sidebarLogo) sidebarLogo.src = logoSrc;
}

// Filtros de Historial
if(btnFilter) btnFilter.addEventListener('click', () => fetchHistory(1));
if(prevBtn) prevBtn.addEventListener('click', () => { if(currentPage > 1) fetchHistory(currentPage - 1); });
if(nextBtn) nextBtn.addEventListener('click', () => fetchHistory(currentPage + 1));


// --- 5. LÓGICA DE GRÁFICAS ---

async function loadChartData(userId) {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 30);

    // Obtener sesiones
    const { data: sesiones } = await supabase
        .from('sesiones_conduccion')
        .select('id_sesion')
        .eq('id_usuario', userId);
    
    const sessionIds = sesiones ? sesiones.map(s => s.id_sesion) : [];

    if (sessionIds.length === 0) {
        renderEmptyChart();
        return;
    }

    // Consulta a la Vista
    const { data: stats, error } = await supabase
        .from('resumen_diario')
        .select('fecha, riesgo, total_minutos')
        .in('id_sesion', sessionIds)
        .gte('fecha', startDate.toISOString())
        .order('fecha', { ascending: true });

    if (error) {
        console.error("Error cargando gráfico:", error);
        return;
    }

    processAndRenderChart(stats, startDate, endDate);
}

function processAndRenderChart(stats, startDate, endDate) {
    const dailyStats = {};
    
    // Rellenar calendario
    for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 1)) {
        const dateStr = d.toISOString().split('T')[0];
        dailyStats[dateStr] = { Normal: 0, Leve: 0, Moderado: 0, Alto: 0, Total: 0 };
    }

    stats.forEach(item => {
        const dateStr = item.fecha; 
        if (dailyStats[dateStr]) {
            let nivel = item.riesgo || 'Normal';
            if (nivel === 'Alto riesgo') nivel = 'Alto';
            
            if (dailyStats[dateStr][nivel] !== undefined) {
                dailyStats[dateStr][nivel] += item.total_minutos; 
                dailyStats[dateStr].Total += item.total_minutos;
            }
        }
    });

    const labels = Object.keys(dailyStats);
    const dataNormal = [], dataLeve = [], dataModerado = [], dataAlto = [];

    labels.forEach(date => {
        const day = dailyStats[date];
        const total = day.Total || 1; 

        if (day.Total === 0) {
            dataNormal.push(null); 
            dataLeve.push(null);
            dataModerado.push(null);
            dataAlto.push(null);
        } else {
            dataNormal.push((day.Normal / total) * 100);
            dataLeve.push((day.Leve / total) * 100);
            dataModerado.push((day.Moderado / total) * 100);
            dataAlto.push((day.Alto / total) * 100);
        }
    });

    renderChart(labels, dataNormal, dataLeve, dataModerado, dataAlto);
}

function renderChart(labels, dNormal, dLeve, dModerado, dAlto) {
    const ctx = document.getElementById('fatigueChart').getContext('2d');
    const isMobile = window.innerWidth < 768;
    
    // Detectar tema actual para colores iniciales
    const isLight = document.body.getAttribute('data-theme') === 'light';
    const textColor = isLight ? '#475569' : '#94a3b8';
    
    if (myChart) myChart.destroy();

    myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Normal',
                    data: dNormal,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.15)',
                    borderWidth: isMobile ? 1.5 : 2,
                    tension: 0.3,
                    pointRadius: isMobile ? 2 : 3,
                    pointHoverRadius: 5,
                    pointHitRadius: 40,
                    fill: true 
                },
                {
                    label: 'Fatiga',
                    data: dLeve,
                    borderColor: '#facc15',
                    backgroundColor: 'rgba(250, 204, 21, 0.2)',
                    borderWidth: isMobile ? 1.5 : 2,
                    tension: 0.3,
                    pointRadius: isMobile ? 2 : 3,
                    pointHoverRadius: 5,
                    pointHitRadius: 40,
                    fill: true
                },
                {
                    label: 'Somnolencia',
                    data: dModerado,
                    borderColor: '#f97316',
                    backgroundColor: 'rgba(249, 115, 22, 0.2)',
                    borderWidth: isMobile ? 1.5 : 2,
                    tension: 0.3,
                    pointRadius: isMobile ? 2 : 3,
                    pointHoverRadius: 5,
                    pointHitRadius: 40,
                    fill: true
                },
                {
                    label: 'Microsueño',
                    data: dAlto,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.2)',
                    borderWidth: isMobile ? 1.5 : 2,
                    tension: 0.3,
                    pointRadius: isMobile ? 2 : 3,
                    pointHoverRadius: 5,
                    pointHitRadius: 40,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            resizeDelay: 200,
            
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            },

            layout: { padding: { top: 20, left: 5, right: 10, bottom: 0 } },
            
            scales: {
                x: {
                    ticks: { 
                        color: textColor, // Variable
                        font: { size: isMobile ? 10 : 11 },
                        minRotation: isMobile ? 0 : 45, 
                        maxRotation: isMobile ? 0 : 45,
                        autoSkip: true, 
                        autoSkipPadding: 15,
                        maxTicksLimit: isMobile ? 12 : 31,
                        callback: function(val) {
                            const label = this.getLabelForValue(val);
                            return isMobile ? label.split('-')[2] + '/' + label.split('-')[1] : label;
                        }
                    },
                    grid: { color: isLight ? '#e2e8f0' : '#334155', drawBorder: false, display: true }
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: { 
                        color: textColor, // Variable
                        font: { size: 10 },
                        callback: v => v + "%", 
                        maxTicksLimit: 6 
                    },
                    grid: { color: isLight ? '#e2e8f0' : '#334155', drawBorder: false },
                    title: { display: !isMobile, text: 'Tiempo (%)', color: textColor }
                }
            },
            plugins: {
                legend: { display: false },
                title: { display: false },
                tooltip: {
                    enabled: true,
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleFont: { size: isMobile ? 11 : 13, weight: 'bold' },
                    bodyFont: { size: isMobile ? 11 : 12 },
                    footerFont: { size: isMobile ? 9 : 11 },
                    padding: isMobile ? 8 : 10,
                    bodySpacing: isMobile ? 3 : 6,
                    boxWidth: isMobile ? 8 : 10,
                    boxHeight: isMobile ? 8 : 10,
                    yAlign: 'bottom', 
                    displayColors: true,
                    callbacks: {
                        title: function(context) { return context[0].label; },
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (context.parsed.y !== null) label += ': ' + context.parsed.y.toFixed(0) + '%';
                            return label;
                        }
                    }
                }
            }
        }
    });

    generateHtmlLegend(myChart);

    if (isMobile) {
        setTimeout(() => {
            const scrollWrapper = document.querySelector('.chart-scroll-wrapper');
            if (scrollWrapper && scrollWrapper.scrollWidth > scrollWrapper.clientWidth) {
                scrollWrapper.scrollLeft = scrollWrapper.scrollWidth;
            }
        }, 100);
    }
}

function generateHtmlLegend(chart) {
    const legendContainer = document.getElementById('customLegend');
    if(!legendContainer) return;
    
    legendContainer.innerHTML = ''; 

    chart.data.datasets.forEach((dataset, index) => {
        const item = document.createElement('div');
        item.className = 'legend-item';
        
        const colorBox = document.createElement('span');
        colorBox.className = 'legend-color-box';
        colorBox.style.backgroundColor = dataset.borderColor;

        const text = document.createElement('span');
        text.textContent = dataset.label;

        item.appendChild(colorBox);
        item.appendChild(text);

        item.onclick = () => {
            const isVisible = chart.isDatasetVisible(index);
            if (isVisible) {
                chart.hide(index);
                item.classList.add('hidden');
            } else {
                chart.show(index);
                item.classList.remove('hidden');
            }
        };

        legendContainer.appendChild(item);
    });
}

function renderEmptyChart() {
    const ctx = document.getElementById('fatigueChart').getContext('2d');
    if (myChart) myChart.destroy();
    myChart = new Chart(ctx, {
        type: 'line',
        data: { labels: ['Sin datos'], datasets: [{ label: 'Sin actividad', data: [], borderColor: '#334155' }] },
        options: { responsive: true, plugins: { legend: { display: false } } }
    });
}


// --- 6. LÓGICA DE HISTORIAL ---

async function fetchHistory(page) {
    if (!currentUserId) return;
    if(tableBody) tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center;">Cargando registros...</td></tr>';
    
    const fDate = document.getElementById('filterDate').value;
    const fType = document.getElementById('filterType').value;
    const fLevel = document.getElementById('filterLevel').value;
    const from = (page - 1) * pageSize;
    const to = from + pageSize - 1;

    const { data: sesiones } = await supabase.from('sesiones_conduccion').select('id_sesion').eq('id_usuario', currentUserId);
    const sessionIds = sesiones ? sesiones.map(s => s.id_sesion) : [];

    if (sessionIds.length === 0) { renderTable([], 0); return; }

    let query = supabase.from('Alertas').select('*', { count: 'exact' }).in('id_sesion', sessionIds).order('fecha_alerta', { ascending: false }).range(from, to);

    if (fLevel) query = query.eq('nivel_riesgo', fLevel);
    if (fType) query = query.ilike('causa_detonante', `%${fType}%`);
    if (fDate) {
        const startDate = new Date(fDate);
        const endDate = new Date(fDate);
        endDate.setDate(endDate.getDate() + 1);
        query = query.gte('fecha_alerta', startDate.toISOString()).lt('fecha_alerta', endDate.toISOString());
    }

    const { data, count, error } = await query;
    if (!error) { currentPage = page; renderTable(data, count); }
}

function renderTable(data, totalCount) {
    if(!tableBody) return;
    tableBody.innerHTML = '';
    
    if (!data || data.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center;">No se encontraron alertas registradas.</td></tr>';
        pageInfo.textContent = `Página 1 de 1`;
        return;
    }
    
    data.forEach(alerta => {
        const row = document.createElement('tr');
        const fecha = new Date(alerta.fecha_alerta).toLocaleString('es-ES', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
        let levelClass = 'level-normal';
        if (alerta.nivel_riesgo === 'Alto riesgo') levelClass = 'level-alto';
        if (alerta.nivel_riesgo === 'Moderado') levelClass = 'level-moderado';
        if (alerta.nivel_riesgo === 'Leve') levelClass = 'level-leve';

        let causaMostrar = alerta.causa_detonante || 'Desconocida';
        let valorDisplay = alerta.valor_medido ? alerta.valor_medido.toString() : '-';

        if (causaMostrar.includes("Somnolencia")) valorDisplay += " eventos"; 
        else if (causaMostrar.includes("Microsueño")) valorDisplay += " seg";
        else if (causaMostrar.includes("Bostezos")) valorDisplay += " veces";
        else if (causaMostrar.includes("Fatiga")) valorDisplay += " parpadeos/min";

        row.innerHTML = `<td>${fecha}</td><td>${causaMostrar}</td><td class="${levelClass}">${alerta.nivel_riesgo}</td><td>${valorDisplay}</td>`;
        tableBody.appendChild(row);
    });

    const totalPages = Math.ceil(totalCount / pageSize);
    if(pageInfo) pageInfo.textContent = `Página ${currentPage} de ${totalPages || 1}`;
    
    if(prevBtn) {
        prevBtn.disabled = currentPage === 1;
        prevBtn.style.opacity = currentPage === 1 ? "0.5" : "1";
    }
    if(nextBtn) {
        nextBtn.disabled = currentPage >= totalPages;
        nextBtn.style.opacity = currentPage >= totalPages ? "0.5" : "1";
    }
}

// Iniciar aplicación
initDashboard();