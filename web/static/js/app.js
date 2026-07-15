/**
 * contentValidationAgent — frontend controller.
 *
 * Vanilla ES6, no framework, no build step. One class owns all DOM state
 * so event handlers don't reach into globals — everything routes through
 * `this`. Loaded as a <script type="module"> so top-level `const`/class
 * declarations don't leak into global scope.
 */

class ContentAgentUI {
  constructor() {
    this.mode = "topic"; // "topic" | "document"
    this.selectedFile = null;
    this.isGenerating = false;

    this.cacheElements();
    this.bindEvents();
  }

  cacheElements() {
    this.el = {
      modeTopicBtn: document.getElementById("mode-topic-btn"),
      modeDocBtn: document.getElementById("mode-doc-btn"),
      topicGroup: document.getElementById("topic-input-group"),
      docGroup: document.getElementById("doc-input-group"),
      topicInput: document.getElementById("topic-input"),
      docInput: document.getElementById("doc-input"),
      fileDropText: document.getElementById("file-drop-text"),

      platformSelect: document.getElementById("platform-select"),

      generateBtn: document.getElementById("generate-btn"),
      generateBtnLabel: document.getElementById("generate-btn-label"),

      statusDot: document.getElementById("status-dot"),
      statusText: document.getElementById("status-text"),
      statusList: document.getElementById("status-list"),

      emptyState: document.getElementById("empty-state"),
      result: document.getElementById("result"),
      draftBox: document.getElementById("draft-box"),
      voiceScoreValue: document.getElementById("voice-score-value"),
      aiScoreValue: document.getElementById("ai-score-value"),
      issuesList: document.getElementById("issues-list"),

      editBtn: document.getElementById("edit-btn"),
      approveBtn: document.getElementById("approve-btn"),
    };
  }

  bindEvents() {
    this.el.modeTopicBtn.addEventListener("click", () => this.setMode("topic"));
    this.el.modeDocBtn.addEventListener("click", () => this.setMode("document"));
    this.el.docInput.addEventListener("change", (e) => this.handleFileSelect(e));
    this.el.generateBtn.addEventListener("click", () => this.handleGenerate());
    this.el.editBtn.addEventListener("click", () => this.toggleEditMode());
    this.el.approveBtn.addEventListener("click", () => this.handleApprove());
  }

  setMode(mode) {
    this.mode = mode;
    const isTopic = mode === "topic";

    this.el.modeTopicBtn.classList.toggle("is-active", isTopic);
    this.el.modeTopicBtn.setAttribute("aria-selected", String(isTopic));
    this.el.modeDocBtn.classList.toggle("is-active", !isTopic);
    this.el.modeDocBtn.setAttribute("aria-selected", String(!isTopic));

    this.el.topicGroup.classList.toggle("hidden", !isTopic);
    this.el.docGroup.classList.toggle("hidden", isTopic);
  }

  handleFileSelect(event) {
    const [file] = event.target.files;
    this.selectedFile = file ?? null;
    this.el.fileDropText.textContent = file
      ? file.name
      : "Choose a .docx, .txt, or .md file";
  }

  setPipelineStep(stepName) {
    const steps = ["generate", "validate", "review"];
    const currentIndex = steps.indexOf(stepName);

    this.el.statusList.querySelectorAll(".status-item").forEach((item) => {
      const itemStep = item.dataset.step;
      const itemIndex = steps.indexOf(itemStep);
      const icon = item.querySelector("i");

      item.classList.remove("is-done", "is-active");
      icon.className = "ti ti-circle-dashed";

      if (itemIndex < currentIndex) {
        item.classList.add("is-done");
        icon.className = "ti ti-circle-check";
      } else if (itemIndex === currentIndex) {
        item.classList.add("is-active");
        icon.className = "ti ti-loader-2";
      }
    });
  }

  setBusy(isBusy) {
    this.isGenerating = isBusy;
    this.el.generateBtn.disabled = isBusy;
    this.el.generateBtnLabel.textContent = isBusy ? "Generating…" : "Generate draft";

    this.el.statusDot.classList.toggle("is-busy", isBusy);
    this.el.statusDot.classList.toggle("is-ready", !isBusy);
    this.el.statusText.textContent = isBusy ? "running" : "idle";
  }

  buildRequestBody() {
    const formData = new FormData();
    formData.append("platform", this.el.platformSelect.value);

    if (this.mode === "document" && this.selectedFile) {
      formData.append("document", this.selectedFile);
    } else {
      formData.append("topic", this.el.topicInput.value.trim());
    }

    return formData;
  }

  validateInput() {
    if (this.mode === "topic" && !this.el.topicInput.value.trim()) {
      return "Enter a topic first.";
    }
    if (this.mode === "document" && !this.selectedFile) {
      return "Choose a document first.";
    }
    return null;
  }

  async handleGenerate() {
    if (this.isGenerating) return;

    const validationError = this.validateInput();
    if (validationError) {
      alert(validationError);
      return;
    }

    this.setBusy(true);
    this.setPipelineStep("generate");
    this.el.emptyState.classList.add("hidden");
    this.el.result.classList.add("hidden");

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        body: this.buildRequestBody(),
      });

      this.setPipelineStep("validate");

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error ?? "Something went wrong.");
      }

      this.setPipelineStep(data.needs_human_review ? "review" : "validate");
      this.renderResult(data);
    } catch (error) {
      this.renderError(error.message);
    } finally {
      this.setBusy(false);
    }
  }

  renderResult(data) {
    this.el.result.classList.remove("hidden");
    this.el.draftBox.textContent = data.draft;

    this.el.voiceScoreValue.textContent = data.voice_score.toFixed(2);
    this.el.aiScoreValue.textContent = data.ai_sounding_score.toFixed(2);

    this.renderIssues(data.issues);
  }

  renderIssues(issues) {
    this.el.issuesList.innerHTML = "";

    if (!issues || issues.length === 0) return;

    const iconByCategory = {
      voice: "ti-microphone",
      ai_sounding: "ti-robot",
      platform_fit: "ti-layout",
      factual_flag: "ti-alert-triangle",
      length: "ti-text-wrap",
    };

    issues.forEach((issue) => {
      const item = document.createElement("div");
      item.className = `issue-item severity-${issue.severity}`;

      const icon = document.createElement("i");
      icon.className = `ti ${iconByCategory[issue.category] ?? "ti-info-circle"}`;
      icon.setAttribute("aria-hidden", "true");

      const text = document.createElement("span");
      text.textContent = issue.message;

      item.append(icon, text);
      this.el.issuesList.appendChild(item);
    });
  }

  renderError(message) {
    this.el.result.classList.remove("hidden");
    this.el.draftBox.textContent = `Error: ${message}`;
    this.el.voiceScoreValue.textContent = "—";
    this.el.aiScoreValue.textContent = "—";
    this.el.issuesList.innerHTML = "";
  }

  toggleEditMode() {
    const isEditable = this.el.draftBox.getAttribute("contenteditable") === "true";
    this.el.draftBox.setAttribute("contenteditable", String(!isEditable));
    this.el.editBtn.textContent = isEditable ? "Edit" : "Done editing";
    if (!isEditable) this.el.draftBox.focus();
  }

  async handleApprove() {
    const text = this.el.draftBox.textContent;

    try {
      await navigator.clipboard.writeText(text);
      const original = this.el.approveBtn.textContent;
      this.el.approveBtn.textContent = "Copied";
      setTimeout(() => {
        this.el.approveBtn.textContent = original;
      }, 1500);
    } catch {
      // Clipboard API can fail without HTTPS/permissions — fall back
      // silently rather than throwing in the console for a non-critical action.
      alert("Copy this manually:\n\n" + text);
    }
  }
}

document.addEventListener("DOMContentLoaded", () => new ContentAgentUI());
