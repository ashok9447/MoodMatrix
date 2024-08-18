document.addEventListener('DOMContentLoaded', function() {
    var flashMessages = document.querySelectorAll('.alert');

    flashMessages.forEach(function(message) {
        setTimeout(function() {
            message.style.display = 'none';
        }, 3000); 
    });
});
