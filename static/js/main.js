document.addEventListener("DOMContentLoaded", () => {
  const header = document.querySelector("[data-header]");
  const navToggle = document.querySelector("[data-nav-toggle]");
  const navLinks = document.querySelector("[data-nav-links]");

  const updateHeader = () => header?.classList.toggle("scrolled", window.scrollY > 12);
  updateHeader();
  window.addEventListener("scroll", updateHeader, { passive: true });

  navToggle?.addEventListener("click", () => {
    const open = navToggle.getAttribute("aria-expanded") === "true";
    navToggle.setAttribute("aria-expanded", String(!open));
    navToggle.setAttribute("aria-label", open ? "Open navigation" : "Close navigation");
    navLinks?.classList.toggle("open", !open);
  });
  navLinks?.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => {
    navLinks.classList.remove("open");
    navToggle?.setAttribute("aria-expanded", "false");
  }));

  document.querySelectorAll("[data-year]").forEach((node) => { node.textContent = new Date().getFullYear(); });
  document.querySelectorAll("[data-tech-logo]").forEach((logo) => {
    logo.addEventListener("error", () => { logo.hidden = true; });
  });
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) { entry.target.classList.add("visible"); revealObserver.unobserve(entry.target); }
    });
  }, { threshold: 0.08 });
  document.querySelectorAll(".reveal").forEach((node) => revealObserver.observe(node));

  initArchitectureModal();
  initAnalyzer();
});

function initArchitectureModal() {
  const modal = document.querySelector("[data-architecture-modal]");
  const openButton = document.querySelector("[data-architecture-open]");
  const closeButton = modal?.querySelector("[data-architecture-close]");
  if (!modal || !openButton || !closeButton) return;

  openButton.addEventListener("click", () => modal.showModal());
  closeButton.addEventListener("click", () => modal.close());
  modal.addEventListener("click", (event) => {
    if (event.target === modal) modal.close();
  });
}

function initAnalyzer() {
  const analyzer = document.querySelector("[data-analyzer]");
  if (!analyzer) return;

  const input = analyzer.querySelector("[data-file-input]");
  const zone = analyzer.querySelector("[data-upload-zone]");
  const empty = analyzer.querySelector("[data-upload-empty]");
  const previewWrap = analyzer.querySelector("[data-preview-wrap]");
  const preview = analyzer.querySelector("[data-preview]");
  const fileInfo = analyzer.querySelector("[data-file-info]");
  const fileName = analyzer.querySelector("[data-file-name]");
  const fileSize = analyzer.querySelector("[data-file-size]");
  const formMessage = analyzer.querySelector("[data-form-message]");
  const analyzeButton = analyzer.querySelector("[data-analyze]");
  const analyzeText = analyzer.querySelector("[data-analyze-text]");
  const states = {
    awaiting: analyzer.querySelector("[data-result-state]"),
    scanning: analyzer.querySelector("[data-scanning]"),
    completed: analyzer.querySelector("[data-completed]"),
    failed: analyzer.querySelector("[data-failed]")
  };
  let selectedFile = null;
  let previewUrl = null;
  let running = false;

  const showState = (name) => Object.entries(states).forEach(([key, node]) => { node.hidden = key !== name; });
  const showMessage = (message) => { formMessage.textContent = message; };
  const openPicker = () => { if (!running) input.click(); };

  function validate(file) {
    if (!file) return "Please select a chest X-ray image before analyzing.";
    const supportedTypes = ["image/jpeg", "image/png"];
    const extensionOkay = /\.(jpe?g|png)$/i.test(file.name);
    if (!supportedTypes.includes(file.type) || !extensionOkay) return "Unsupported format. Please choose a JPEG or PNG image.";
    if (file.size > 10 * 1024 * 1024) return "This image is larger than 10 MB. Please choose a smaller file.";
    return "";
  }

  function selectFile(file) {
    const error = validate(file);
    if (error) { showMessage(error); return; }
    selectedFile = file;
    analyzer.classList.add("has-file");
    showMessage("");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    empty.hidden = true;
    previewWrap.hidden = false;
    fileInfo.hidden = false;
    fileName.textContent = file.name;
    fileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB · ${file.type === "image/png" ? "PNG" : "JPEG"}`;
    analyzeButton.disabled = false;
    showState("awaiting");
    states.awaiting.querySelector("p").textContent = "Image ready. Click Analyze X-Ray to run the classification model.";
  }

  function reset() {
    selectedFile = null;
    analyzer.classList.remove("has-file");
    input.value = "";
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    preview.src = "";
    empty.hidden = false;
    previewWrap.hidden = true;
    fileInfo.hidden = true;
    analyzeButton.disabled = true;
    analyzeText.textContent = "Analyze X-Ray";
    states.completed.classList.remove("attention", "normal");
    showMessage("");
    showState("awaiting");
    states.awaiting.querySelector("p").textContent = "Select a supported image and start the analysis. Results from the classification model will appear here.";
  }

  zone.addEventListener("click", (event) => { if (!event.target.closest("button")) openPicker(); });
  zone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); openPicker(); } });
  analyzer.querySelector("[data-choose-file]").addEventListener("click", openPicker);
  analyzer.querySelector("[data-change-file]").addEventListener("click", openPicker);
  analyzer.querySelector("[data-remove-file]").addEventListener("click", reset);
  analyzer.querySelector("[data-reset]").addEventListener("click", reset);
  analyzer.querySelector("[data-retry]").addEventListener("click", () => runAnalysis());
  input.addEventListener("change", () => selectFile(input.files[0]));

  ["dragenter", "dragover"].forEach((eventName) => zone.addEventListener(eventName, (event) => { event.preventDefault(); zone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((eventName) => zone.addEventListener(eventName, (event) => { event.preventDefault(); zone.classList.remove("dragover"); }));
  zone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

  async function runAnalysis() {
    if (running) return;
    const error = validate(selectedFile);
    if (error) { showMessage(error); return; }
    running = true;
    showMessage("");
    analyzeButton.disabled = true;
    analyzeText.textContent = "Analyzing...";
    showState("scanning");

    const loadingMessage = analyzer.querySelector("[data-loading-message]");
    const messages = ["Preparing radiograph...", "Applying inference preprocessing...", "Running custom CNN...", "Generating classification..."];
    let messageIndex = 0;
    loadingMessage.textContent = messages[0];
    const messageTimer = window.setInterval(() => { messageIndex = Math.min(messageIndex + 1, messages.length - 1); loadingMessage.textContent = messages[messageIndex]; }, 650);
    const startedAt = Date.now();
    const requestController = new AbortController();
    const requestTimeout = window.setTimeout(() => requestController.abort(), 30000);

    try {
      const body = new FormData();
      body.append("file", selectedFile);
      const response = await fetch("/predict", { method: "POST", body, signal: requestController.signal });
      let data = null;
      try { data = await response.json(); } catch (_) { /* handled below */ }
      if (!response.ok) throw new Error(data?.detail || `Prediction service returned ${response.status}.`);
      if (!data || !["NORMAL", "PNEUMONIA"].includes(data.prediction_label) || ![0, 1].includes(data.prediction_index)) throw new Error("The prediction service returned an unexpected response.");

      const remaining = Math.max(0, 1700 - (Date.now() - startedAt));
      await new Promise((resolve) => window.setTimeout(resolve, remaining));
      displayResult(data.prediction_label);
    } catch (predictionError) {
      analyzer.querySelector("[data-error-message]").textContent = predictionError.name === "AbortError"
        ? "The prediction service did not respond within 30 seconds. Please verify the FastAPI server and try again."
        : predictionError.message || "The prediction could not be completed. Please check the server and try again.";
      showState("failed");
    } finally {
      window.clearTimeout(requestTimeout);
      window.clearInterval(messageTimer);
      running = false;
      analyzeButton.disabled = !selectedFile;
      analyzeText.textContent = "Analyze X-Ray";
    }
  }

  function displayResult(label) {
    const normal = label === "NORMAL";
    states.completed.classList.toggle("normal", normal);
    states.completed.classList.toggle("attention", !normal);
    analyzer.querySelector("[data-result-icon]").textContent = normal ? "✓" : "!";
    analyzer.querySelector("[data-result-label]").textContent = `Model Classification: ${label}`;
    analyzer.querySelector("[data-result-description]").textContent = normal
      ? "The model classified this image as NORMAL. This result is informational and requires professional interpretation."
      : "The model classified this image as PNEUMONIA. This is not a diagnosis; professional medical interpretation is recommended.";
    showState("completed");
  }

  analyzeButton.addEventListener("click", runAnalysis);
}
