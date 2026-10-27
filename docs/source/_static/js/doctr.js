// Copy-to-clipboard for the install command of the landing page
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(button.dataset.copy);
        const label = button.textContent;
        button.textContent = "Copied!";
        setTimeout(() => (button.textContent = label), 1500);
      } catch (err) {
        console.warn("[docTR] unable to copy to the clipboard", err);
      }
    });
  });
});
