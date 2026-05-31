(function () {
  "use strict";

  var ENHANCED_ATTR = "data-diagram-viewer-enhanced";
  var RENDERING_ATTR = "data-diagram-viewer-rendering";
  var SOURCE_ATTR = "data-diagram-viewer-source";
  var MODAL_ID = "diagram-viewer";
  var MIN_ZOOM = 0.5;
  var MAX_ZOOM = 4;
  var ZOOM_STEP = 0.25;
  var RETRY_DELAYS = [0, 100, 300, 700, 1500, 3000];

  var modal = null;
  var viewport = null;
  var stage = null;
  var title = null;
  var zoomLabel = null;
  var activeSvg = null;
  var activeOpener = null;
  var activeBaseWidth = 720;
  var zoom = 1;
  var isPanning = false;
  var panStartX = 0;
  var panStartY = 0;
  var panStartLeft = 0;
  var panStartTop = 0;
  var observer = null;
  var enhanceTimer = null;
  var renderIndex = 0;
  var modalSvgIndex = 0;
  var mermaidInitialized = false;

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function getDiagramLabel(diagram) {
    var heading = null;
    var cursor = diagram.previousElementSibling;

    while (cursor) {
      if (/^H[1-6]$/.test(cursor.tagName)) {
        heading = cursor;
        break;
      }
      cursor = cursor.previousElementSibling;
    }

    if (heading && heading.textContent.trim()) {
      var headingClone = heading.cloneNode(true);
      headingClone.querySelectorAll(".headerlink").forEach(function (link) {
        link.remove();
      });

      return headingClone.textContent.replace(/\s+/g, " ").trim();
    }

    return "Diagram";
  }

  function getRenderedSvg(diagram) {
    return diagram.querySelector("svg");
  }

  function getDiagramSource(diagram) {
    var storedSource = diagram.getAttribute(SOURCE_ATTR);

    if (storedSource) {
      return storedSource;
    }

    var code = diagram.querySelector("code");
    var source = (code || diagram).textContent.trim();

    if (source) {
      diagram.setAttribute(SOURCE_ATTR, source);
    }

    return source;
  }

  function normalizeDiagramElement(diagram) {
    var normalized = diagram;

    if (diagram.tagName === "PRE") {
      normalized = document.createElement("div");
      normalized.className = diagram.className || "mermaid";
      normalized.textContent = getDiagramSource(diagram);
      diagram.replaceWith(normalized);
    }

    if (!normalized.classList.contains("mermaid")) {
      normalized.classList.add("mermaid");
    }

    return normalized;
  }

  function getViewBoxWidth(svg) {
    var viewBox = svg.getAttribute("viewBox");

    if (!viewBox) {
      return 0;
    }

    var parts = viewBox
      .trim()
      .split(/[\s,]+/)
      .map(function (part) {
        return Number(part);
      });

    return parts.length === 4 && Number.isFinite(parts[2]) ? parts[2] : 0;
  }

  function getBaseWidth(sourceSvg) {
    var sourceRect = sourceSvg.getBoundingClientRect();
    var sourceWidth = sourceRect.width || Number(sourceSvg.getAttribute("width")) || 0;
    var viewportWidth = viewport ? viewport.clientWidth - 32 : 0;
    var viewBoxWidth = getViewBoxWidth(sourceSvg);

    return Math.max(640, sourceWidth, viewportWidth, viewBoxWidth);
  }

  function replaceSvgIdReference(value, sourceId, modalId) {
    return value && sourceId ? value.split(sourceId).join(modalId) : value;
  }

  function cloneSvgForModal(sourceSvg) {
    var clone = sourceSvg.cloneNode(true);
    var sourceId = sourceSvg.getAttribute("id");
    var modalId = sourceId ? sourceId + "-modal-" + modalSvgIndex++ : "";

    clone.removeAttribute("style");
    clone.setAttribute("aria-hidden", "true");
    clone.setAttribute("focusable", "false");
    clone.classList.add("diagram-viewer__svg");

    if (!sourceId) {
      return clone;
    }

    clone.setAttribute("id", modalId);
    clone.querySelectorAll("style").forEach(function (style) {
      style.textContent = replaceSvgIdReference(style.textContent, sourceId, modalId);
    });
    clone.querySelectorAll("*").forEach(function (element) {
      Array.prototype.slice.call(element.attributes).forEach(function (attribute) {
        if (attribute.value.indexOf(sourceId) !== -1) {
          element.setAttribute(
            attribute.name,
            replaceSvgIdReference(attribute.value, sourceId, modalId)
          );
        }
      });
    });

    return clone;
  }

  function ensureModal() {
    if (modal) {
      return modal;
    }

    modal = document.createElement("div");
    modal.id = MODAL_ID;
    modal.className = "diagram-viewer";
    modal.hidden = true;
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.setAttribute("aria-labelledby", "diagram-viewer-title");

    modal.innerHTML = [
      '<div class="diagram-viewer__backdrop" data-diagram-viewer-close></div>',
      '<div class="diagram-viewer__dialog">',
      '  <div class="diagram-viewer__toolbar">',
      '    <h2 class="diagram-viewer__title" id="diagram-viewer-title">Diagram</h2>',
      '    <div class="diagram-viewer__controls" aria-label="Diagram controls">',
      '      <button class="diagram-viewer__button" type="button" data-diagram-viewer-zoom-out aria-label="Zoom out">-</button>',
      '      <span class="diagram-viewer__zoom" aria-live="polite">100%</span>',
      '      <button class="diagram-viewer__button" type="button" data-diagram-viewer-zoom-in aria-label="Zoom in">+</button>',
      '      <button class="diagram-viewer__button diagram-viewer__button--wide" type="button" data-diagram-viewer-reset>Reset</button>',
      '      <button class="diagram-viewer__button diagram-viewer__button--wide" type="button" data-diagram-viewer-close>Close</button>',
      "    </div>",
      "  </div>",
      '  <div class="diagram-viewer__viewport" tabindex="0">',
      '    <div class="diagram-viewer__stage"></div>',
      "  </div>",
      "</div>",
    ].join("");

    document.body.appendChild(modal);

    viewport = modal.querySelector(".diagram-viewer__viewport");
    stage = modal.querySelector(".diagram-viewer__stage");
    title = modal.querySelector(".diagram-viewer__title");
    zoomLabel = modal.querySelector(".diagram-viewer__zoom");

    modal
      .querySelector("[data-diagram-viewer-zoom-out]")
      .addEventListener("click", function () {
        setZoom(zoom - ZOOM_STEP);
      });
    modal
      .querySelector("[data-diagram-viewer-zoom-in]")
      .addEventListener("click", function () {
        setZoom(zoom + ZOOM_STEP);
      });
    modal
      .querySelector("[data-diagram-viewer-reset]")
      .addEventListener("click", function () {
        setZoom(1);
        centerViewport();
      });

    modal.addEventListener("click", function (event) {
      if (
        event.target instanceof Element &&
        event.target.closest("[data-diagram-viewer-close]")
      ) {
        closeModal();
      }
    });

    viewport.addEventListener("pointerdown", beginPan);
    viewport.addEventListener("pointermove", movePan);
    viewport.addEventListener("pointerup", endPan);
    viewport.addEventListener("pointercancel", endPan);
    viewport.addEventListener("lostpointercapture", endPan);

    document.addEventListener("keydown", handleKeydown);

    return modal;
  }

  function setZoom(nextZoom) {
    zoom = clamp(nextZoom, MIN_ZOOM, MAX_ZOOM);
    zoomLabel.textContent = Math.round(zoom * 100) + "%";

    if (activeSvg) {
      activeSvg.style.width = activeBaseWidth * zoom + "px";
      activeSvg.style.maxWidth = "none";
    }
  }

  function centerViewport() {
    if (!viewport) {
      return;
    }

    requestAnimationFrame(function () {
      viewport.scrollLeft = (viewport.scrollWidth - viewport.clientWidth) / 2;
      viewport.scrollTop = (viewport.scrollHeight - viewport.clientHeight) / 2;
    });
  }

  function openModal(diagram, opener) {
    var sourceSvg = getRenderedSvg(diagram);

    if (!sourceSvg) {
      return;
    }

    ensureModal();

    activeOpener = opener;
    activeSvg = cloneSvgForModal(sourceSvg);

    title.textContent = getDiagramLabel(diagram);
    stage.replaceChildren(activeSvg);

    modal.hidden = false;
    document.body.classList.add("diagram-viewer-is-open");

    activeBaseWidth = getBaseWidth(sourceSvg);
    setZoom(1);
    centerViewport();
    viewport.focus();
  }

  function closeModal() {
    if (!modal || modal.hidden) {
      return;
    }

    modal.hidden = true;
    document.body.classList.remove("diagram-viewer-is-open");
    stage.replaceChildren();
    activeSvg = null;

    if (activeOpener && document.contains(activeOpener)) {
      activeOpener.focus();
    }
    activeOpener = null;
  }

  function beginPan(event) {
    if (!viewport || (typeof event.button === "number" && event.button !== 0)) {
      return;
    }

    isPanning = true;
    panStartX = event.clientX;
    panStartY = event.clientY;
    panStartLeft = viewport.scrollLeft;
    panStartTop = viewport.scrollTop;
    viewport.classList.add("diagram-viewer__viewport--panning");
    viewport.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  function movePan(event) {
    if (!isPanning || !viewport) {
      return;
    }

    viewport.scrollLeft = panStartLeft - (event.clientX - panStartX);
    viewport.scrollTop = panStartTop - (event.clientY - panStartY);
  }

  function endPan(event) {
    if (!isPanning || !viewport) {
      return;
    }

    isPanning = false;
    viewport.classList.remove("diagram-viewer__viewport--panning");

    if (viewport.hasPointerCapture(event.pointerId)) {
      viewport.releasePointerCapture(event.pointerId);
    }
  }

  function handleKeydown(event) {
    if (!modal || modal.hidden) {
      return;
    }

    if (event.key === "Escape") {
      event.preventDefault();
      closeModal();
      return;
    }

    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      setZoom(zoom + ZOOM_STEP);
      return;
    }

    if (event.key === "-" || event.key === "_") {
      event.preventDefault();
      setZoom(zoom - ZOOM_STEP);
      return;
    }

    if (event.key === "0") {
      event.preventDefault();
      setZoom(1);
      centerViewport();
    }
  }

  function enhanceDiagram(diagram) {
    if (diagram.getAttribute(ENHANCED_ATTR) === "true" || !getRenderedSvg(diagram)) {
      return;
    }

    diagram.setAttribute(ENHANCED_ATTR, "true");
    diagram.classList.add("diagram-viewer-target");

    var opener = document.createElement("button");
    opener.type = "button";
    opener.className = "diagram-viewer-open";
    opener.textContent = "Open diagram";
    opener.setAttribute("aria-label", "Open diagram in viewer");

    opener.addEventListener("click", function (event) {
      event.stopPropagation();
      openModal(diagram, opener);
    });

    diagram.insertAdjacentElement("beforebegin", opener);

    diagram.addEventListener("click", function () {
      openModal(diagram, opener);
    });

    diagram.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openModal(diagram, opener);
      }
    });

    if (!diagram.hasAttribute("tabindex")) {
      diagram.setAttribute("tabindex", "0");
    }
    if (!diagram.hasAttribute("role")) {
      diagram.setAttribute("role", "button");
    }
    if (!diagram.hasAttribute("aria-label")) {
      diagram.setAttribute("aria-label", "Open diagram in viewer");
    }
  }

  function renderDiagram(diagram) {
    var source = getDiagramSource(diagram);

    if (!source || getRenderedSvg(diagram) || diagram.getAttribute(RENDERING_ATTR) === "true") {
      return;
    }

    initializeMermaid();

    if (!window.mermaid || typeof window.mermaid.render !== "function") {
      return;
    }

    diagram.setAttribute(RENDERING_ATTR, "true");

    window.mermaid
      .render("diagram-viewer-render-" + renderIndex++, source)
      .then(function (result) {
        diagram.innerHTML = result.svg;
        diagram.removeAttribute(RENDERING_ATTR);
        enhanceDiagram(diagram);
      })
      .catch(function () {
        diagram.removeAttribute(RENDERING_ATTR);
      });
  }

  function processDiagram(diagram) {
    var normalized = normalizeDiagramElement(diagram);

    if (getRenderedSvg(normalized)) {
      enhanceDiagram(normalized);
      return;
    }

    renderDiagram(normalized);
  }

  function enhanceDiagrams() {
    ensureModal();
    initializeMermaid();
    document.querySelectorAll(".md-typeset .mermaid").forEach(processDiagram);
  }

  function scheduleEnhance(delay) {
    window.setTimeout(function () {
      if (enhanceTimer) {
        window.clearTimeout(enhanceTimer);
      }

      enhanceTimer = window.setTimeout(enhanceDiagrams, 0);
    }, delay);
  }

  function observeDiagramChanges() {
    if (observer || !document.body || typeof MutationObserver !== "function") {
      return;
    }

    observer = new MutationObserver(function () {
      scheduleEnhance(0);
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }

  function start() {
    ensureModal();
    initializeMermaid();
    observeDiagramChanges();
    RETRY_DELAYS.forEach(scheduleEnhance);
  }

  function initializeMermaid() {
    if (
      mermaidInitialized ||
      !window.mermaid ||
      typeof window.mermaid.initialize !== "function"
    ) {
      return;
    }

    window.mermaid.initialize({ startOnLoad: false });
    mermaidInitialized = true;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(start);
  }
})();
