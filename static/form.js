// Get all seek sliders and seek values
const seekSliders = document.querySelectorAll('input[type="range"]');
const seekValues = document.querySelectorAll('[id^="seek-value"]');

// Iterate through each seek slider and add event listeners
seekSliders.forEach((slider, index) => {
    slider.addEventListener('input', function () {
        seekValues[index].textContent = this.value;
    });
});
