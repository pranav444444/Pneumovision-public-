document.addEventListener("DOMContentLoaded", () => {
  const modal = document.querySelector("[data-training-modal]");
  const openButtons = document.querySelectorAll("[data-training-open]");
  const closeButton = modal?.querySelector("[data-training-close]");
  if (!modal || !openButtons.length || !closeButton) return;

  openButtons.forEach((button) => button.addEventListener("click", () => modal.showModal()));
  closeButton.addEventListener("click", () => modal.close());
  modal.addEventListener("click", (event) => {
    if (event.target === modal) modal.close();
  });
});
