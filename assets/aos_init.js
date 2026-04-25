document.addEventListener('DOMContentLoaded', function () {
    // Wait for the app to render initially
    setTimeout(function () {
        if (typeof AOS !== 'undefined') {
            AOS.init({
                duration: 1000,
                once: true,
                mirror: false
            });
        }
    }, 1000);

    // Watch for new elements added to the DOM (for multi-page Dash apps)
    const observer = new MutationObserver(function (mutations) {
        let shouldRefresh = false;
        for (let mutation of mutations) {
            if (mutation.addedNodes.length > 0) {
                shouldRefresh = true;
                break;
            }
        }
        if (shouldRefresh && typeof AOS !== 'undefined') {
            setTimeout(function () {
                AOS.refreshHard();
            }, 100);
        }
    });

    observer.observe(document.body, { childList: true, subtree: true });
});
