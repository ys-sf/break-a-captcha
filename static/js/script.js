document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const captchaFileInput = document.getElementById('captchaFile');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultDiv = document.getElementById('result');
    const errorMessage = document.getElementById('errorMessage');
    const captchaImage = document.getElementById('captchaImage');
    const captchaText = document.getElementById('captchaText');
    const resetButton = document.getElementById('resetButton');

    // Gestion du formulaire d'upload
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Vérifier si un fichier a été sélectionné
        if (!captchaFileInput.files[0]) {
            showError("Veuillez sélectionner une image CAPTCHA.");
            return;
        }
        
        // Vérifier le type de fichier
        const fileType = captchaFileInput.files[0].type;
        if (!fileType.match('image.*')) {
            showError("Veuillez sélectionner un fichier image valide.");
            return;
        }
        
        // Préparation des données
        const formData = new FormData();
        formData.append('file', captchaFileInput.files[0]);
        
        // Afficher le chargement
        showLoading();
        
        // Envoi de la requête au serveur
        fetch('/api/solve', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || "Une erreur s'est produite");
                });
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                // Afficher le résultat
                showResult(data.image, data.captcha_text);
            } else {
                showError(data.error || "Une erreur s'est produite");
            }
        })
        .catch(error => {
            showError(error.message);
            console.error('Error:', error);
        });
    });
    
    // Gestion du bouton de réinitialisation
    resetButton.addEventListener('click', function() {
        resetForm();
    });
    
    // Fonction pour afficher le chargement
    function showLoading() {
        resetState();
        loadingIndicator.classList.remove('d-none');
    }
    
    // Fonction pour afficher le résultat
    function showResult(imageData, text) {
        loadingIndicator.classList.add('d-none');
        console.log("Image data received:", imageData.substring(0, 50) + "..."); // Log pour débogage
        captchaImage.src = `data:image/png;base64,${imageData}`;
        captchaText.textContent = text;
        resultDiv.classList.remove('d-none');
        resetButton.classList.remove('d-none');
    }
    
    // Fonction pour afficher une erreur
    function showError(message) {
        loadingIndicator.classList.add('d-none');
        errorMessage.textContent = message;
        errorMessage.classList.remove('d-none');
        resetButton.classList.remove('d-none');
    }
    
    // Fonction pour réinitialiser l'état
    function resetState() {
        loadingIndicator.classList.add('d-none');
        resultDiv.classList.add('d-none');
        errorMessage.classList.add('d-none');
        resetButton.classList.add('d-none');
    }
    
    // Fonction pour réinitialiser le formulaire
    function resetForm() {
        uploadForm.reset();
        resetState();
    }
    
    // Animation des cartes de statistiques
    const statCards = document.querySelectorAll('.stat-card');
    statCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.backgroundColor = '#e9ecef';
        });
        card.addEventListener('mouseleave', function() {
            this.style.backgroundColor = '#f8f9fa';
        });
    });
});