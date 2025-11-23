// Traducciones para inglés y español
const translations = {
    es: {
        header: "CERTIFICADO DE EXCELENCIA",
        title: "Certificado de Finalización",
        subtitle: "Este documento certifica que",
        desc: "Este exitoso ha obtenido el certificado de entrenamiento desarrollado en el programa avanzado. Este curso ha sido formado y respaldado en el acto siguiente de estudios en revelaciones sobre su progreso.",
        powered: "IMPULSADO POR TECNOLOGÍAS LÍDERES DE IA"
    },
    en: {
        header: "CERTIFICATE OF EXCELLENCE",
        title: "Certificate of Completion",
        subtitle: "This document certifies that",
        desc: "This successful has certificate obtains training develop in the advanced program. This course is has trained and supported on the act next de studies in revelations on to their progress.",
        powered: "POWERED BY LEADING AI TECHNOLOGIES"
    }
};

// Función para generar el certificado
function generateCertificate() {
    const nameInput = document.getElementById('studentNameInput');
    const name = nameInput.value.trim();
    const lang = document.querySelector('input[name="lang"]:checked').value;
    
    if (!name) {
        nameInput.style.borderColor = '#ff0055';
        nameInput.placeholder = "¡Por favor escribe un nombre!";
        return;
    }

    // Actualizar contenido dinámico
    document.getElementById('displayStudentName').textContent = name;
    document.getElementById('txt-header').textContent = translations[lang].header;
    document.getElementById('txt-title').textContent = translations[lang].title;
    document.getElementById('txt-subtitle').textContent = translations[lang].subtitle;
    document.getElementById('txt-desc').textContent = translations[lang].desc;
    document.getElementById('txt-powered').textContent = translations[lang].powered;

    // Generar fecha actual
    const now = new Date();
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    const dateStr = now.toLocaleDateString(lang === 'es' ? 'es-ES' : 'en-US', options);
    document.getElementById('dateField').innerHTML = (lang === 'es' ? 'FECHA: ' : 'DATE: ') + '<span>' + dateStr + '</span>';

    // Ocultar modal y mostrar controles
    document.getElementById('configModal').classList.add('hidden');
    document.getElementById('uiControls').classList.add('visible');
}

// Función para volver a mostrar el modal
function showModal() {
    document.getElementById('configModal').classList.remove('hidden');
    document.getElementById('uiControls').classList.remove('visible');
}

// Inicializar al cargar la página
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('configModal').classList.remove('hidden');
});
