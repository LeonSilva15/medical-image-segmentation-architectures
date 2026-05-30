(function () {
  const prepared = new WeakSet();
  const MIN_SCALE = 0.05;
  const MAX_SCALE = 8;
  const VIEWPORT_PADDING = 48;

  let active = null;
  let modal = null;
  let viewport = null;
  let stage = null;
  let scaleLabel = null;
  let state = { scale: 1 };
  let pan = null;

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function escapeRegExp(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function parseSvgLength(value) {
    if (!value || value.includes("%")) {
      return null;
    }

    const match = value.match(/^-?\d*\.?\d+/);
    if (!match) {
      return null;
    }

    const number = Number.parseFloat(match[0]);
    return Number.isFinite(number) && number > 0 ? number : null;
  }

  function getSvgViewBox(svg) {
    const value = svg.getAttribute("viewBox");
    if (value) {
      const parts = value
        .trim()
        .split(/[\s,]+/)
        .map((part) => Number.parseFloat(part));

      if (
        parts.length === 4 &&
        parts.every(Number.isFinite) &&
        parts[2] > 0 &&
        parts[3] > 0
      ) {
        return { width: parts[2], height: parts[3] };
      }
    }

    if (svg.viewBox && svg.viewBox.baseVal.width > 0 && svg.viewBox.baseVal.height > 0) {
      return {
        width: svg.viewBox.baseVal.width,
        height: svg.viewBox.baseVal.height,
      };
    }

    return null;
  }

  function getSvgSize(svg) {
    const viewBox = getSvgViewBox(svg);
    if (viewBox) {
      return viewBox;
    }

    const width = parseSvgLength(svg.getAttribute("width"));
    const height = parseSvgLength(svg.getAttribute("height"));
    if (width && height) {
      return { width, height };
    }

    const rect = svg.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      return { width: rect.width, height: rect.height };
    }

    return { width: 800, height: 600 };
  }

  function findRenderedSvg(diagram) {
    if (diagram.tagName && diagram.tagName.toLowerCase() === "svg") {
      return diagram;
    }

    return diagram.querySelector("svg");
  }

  function replaceIdReferences(value, idMap) {
    let updated = value;

    idMap.forEach((newId, oldId) => {
      const escaped = escapeRegExp(oldId);
      updated = updated.replace(
        new RegExp(`url\\(["']?#${escaped}["']?\\)`, "g"),
        `url(#${newId})`
      );
      if (updated === `#${oldId}`) {
        updated = `#${newId}`;
      }
    });

    return updated;
  }

  function prefixSvgIds(svg) {
    const idMap = new Map();
    const prefix = `diagram-viewer-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    [svg, ...svg.querySelectorAll("[id]")].forEach((element) => {
      if (!element.id) {
        return;
      }

      const oldId = element.id;
      const newId = `${prefix}-${oldId}`;
      idMap.set(oldId, newId);
      element.id = newId;
    });

    if (!idMap.size) {
      return;
    }

    [svg, ...svg.querySelectorAll("*")].forEach((element) => {
      Array.from(element.attributes).forEach((attribute) => {
        const updated = replaceIdReferences(attribute.value, idMap);
        if (updated !== attribute.value) {
          element.setAttribute(attribute.name, updated);
        }
      });
    });

    svg.querySelectorAll("style").forEach((style) => {
      style.textContent = replaceIdReferences(style.textContent, idMap);
    });
  }

  function cloneDiagramSvg(diagram) {
    const sourceSvg = findRenderedSvg(diagram);
    if (!sourceSvg) {
      return null;
    }

    const svg = sourceSvg.cloneNode(true);
    svg.classList.add("diagram-viewer__svg");
    svg.removeAttribute("width");
    svg.removeAttribute("height");
    svg.style.width = "";
    svg.style.height = "";
    svg.style.maxWidth = "none";
    svg.style.maxHeight = "none";
    prefixSvgIds(svg);

    return {
      svg,
      size: getSvgSize(sourceSvg),
    };
  }

  function ensureModal() {
    if (modal) {
      return;
    }

    modal = document.createElement("div");
    modal.className = "diagram-viewer__modal";
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.setAttribute("aria-label", "Diagram viewer");
    modal.innerHTML = [
      '<div class="diagram-viewer__backdrop" data-diagram-viewer-close></div>',
      '<div class="diagram-viewer__panel">',
      '  <div class="diagram-viewer__toolbar">',
      '    <div class="diagram-viewer__title">Diagram viewer</div>',
      '    <div class="diagram-viewer__controls">',
      '      <button class="diagram-viewer__control" type="button" data-diagram-viewer-zoom-out aria-label="Zoom out">-</button>',
      '      <button class="diagram-viewer__control" type="button" data-diagram-viewer-zoom-in aria-label="Zoom in">+</button>',
      '      <button class="diagram-viewer__control" type="button" data-diagram-viewer-fit>Fit</button>',
      '      <button class="diagram-viewer__control" type="button" data-diagram-viewer-reset>Reset</button>',
      '      <span class="diagram-viewer__control" aria-live="polite" data-diagram-viewer-scale>100%</span>',
      '      <button class="diagram-viewer__control" type="button" data-diagram-viewer-close>Close</button>',
      "    </div>",
      "  </div>",
      '  <div class="diagram-viewer__stage">',
      '    <div class="diagram-viewer__viewport"></div>',
      "  </div>",
      "</div>",
    ].join("");

    document.body.appendChild(modal);

    viewport = modal.querySelector(".diagram-viewer__viewport");
    stage = modal.querySelector(".diagram-viewer__stage");
    scaleLabel = modal.querySelector("[data-diagram-viewer-scale]");

    modal.querySelectorAll("[data-diagram-viewer-close]").forEach((button) => {
      button.addEventListener("click", closeViewer);
    });
    modal.querySelector("[data-diagram-viewer-fit]").addEventListener("click", fitDiagram);
    modal.querySelector("[data-diagram-viewer-reset]").addEventListener("click", resetDiagram);
    modal.querySelector("[data-diagram-viewer-zoom-in]").addEventListener("click", () => zoomBy(1.2));
    modal.querySelector("[data-diagram-viewer-zoom-out]").addEventListener("click", () => zoomBy(1 / 1.2));

    stage.addEventListener("wheel", onWheel, { passive: false });
    stage.addEventListener("pointerdown", onPointerDown);
    stage.addEventListener("pointermove", onPointerMove);
    stage.addEventListener("pointerup", endPan);
    stage.addEventListener("pointercancel", endPan);

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && active) {
        closeViewer();
      }
    });
  }

  function getLayout(scale) {
    const scaledWidth = active.size.width * scale;
    const scaledHeight = active.size.height * scale;
    const viewportWidth = Math.max(stage.clientWidth, scaledWidth + VIEWPORT_PADDING * 2);
    const viewportHeight = Math.max(stage.clientHeight, scaledHeight + VIEWPORT_PADDING * 2);

    return {
      scaledWidth,
      scaledHeight,
      viewportWidth,
      viewportHeight,
      offsetX: (viewportWidth - scaledWidth) / 2,
      offsetY: (viewportHeight - scaledHeight) / 2,
    };
  }

  function applyLayout(layout) {
    viewport.style.width = `${layout.viewportWidth}px`;
    viewport.style.height = `${layout.viewportHeight}px`;
    active.svg.style.width = `${layout.scaledWidth}px`;
    active.svg.style.height = `${layout.scaledHeight}px`;
    scaleLabel.textContent = `${Math.round(state.scale * 100)}%`;
  }

  function centerDiagram() {
    stage.scrollLeft = (stage.scrollWidth - stage.clientWidth) / 2;
    stage.scrollTop = (stage.scrollHeight - stage.clientHeight) / 2;
  }

  function setScale(nextScale, options) {
    if (!active || !Number.isFinite(nextScale)) {
      return;
    }

    const previousScale = state.scale;
    const previousLayout = getLayout(previousScale);
    const stageRect = stage.getBoundingClientRect();
    const hasAnchor =
      options && Number.isFinite(options.clientX) && Number.isFinite(options.clientY);
    const anchorX = hasAnchor ? options.clientX - stageRect.left : stageRect.width / 2;
    const anchorY = hasAnchor ? options.clientY - stageRect.top : stageRect.height / 2;
    const contentX = (stage.scrollLeft + anchorX - previousLayout.offsetX) / previousScale;
    const contentY = (stage.scrollTop + anchorY - previousLayout.offsetY) / previousScale;

    state.scale = clamp(nextScale, MIN_SCALE, MAX_SCALE);
    const nextLayout = getLayout(state.scale);
    applyLayout(nextLayout);

    if (options && options.center) {
      centerDiagram();
      return;
    }

    stage.scrollLeft = contentX * state.scale + nextLayout.offsetX - anchorX;
    stage.scrollTop = contentY * state.scale + nextLayout.offsetY - anchorY;
  }

  function fitDiagram() {
    if (!active) {
      return;
    }

    const widthRatio = Math.max(stage.clientWidth - VIEWPORT_PADDING * 2, 1) / active.size.width;
    const heightRatio = Math.max(stage.clientHeight - VIEWPORT_PADDING * 2, 1) / active.size.height;
    const fitScale = clamp(Math.min(widthRatio, heightRatio), MIN_SCALE, MAX_SCALE);
    setScale(fitScale, { center: true });
  }

  function resetDiagram() {
    setScale(1, { center: true });
  }

  function zoomBy(factor) {
    setScale(state.scale * factor);
  }

  function onWheel(event) {
    if (!active) {
      return;
    }

    event.preventDefault();
    setScale(state.scale * (event.deltaY < 0 ? 1.12 : 1 / 1.12), {
      clientX: event.clientX,
      clientY: event.clientY,
    });
  }

  function onPointerDown(event) {
    if (!active || event.button !== 0) {
      return;
    }

    event.preventDefault();
    pan = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      scrollLeft: stage.scrollLeft,
      scrollTop: stage.scrollTop,
    };
    stage.classList.add("is-panning");
    stage.setPointerCapture(event.pointerId);
  }

  function onPointerMove(event) {
    if (!pan || event.pointerId !== pan.pointerId) {
      return;
    }

    stage.scrollLeft = pan.scrollLeft - (event.clientX - pan.startX);
    stage.scrollTop = pan.scrollTop - (event.clientY - pan.startY);
  }

  function endPan(event) {
    if (!pan || event.pointerId !== pan.pointerId) {
      return;
    }

    if (stage.hasPointerCapture(event.pointerId)) {
      stage.releasePointerCapture(event.pointerId);
    }

    pan = null;
    stage.classList.remove("is-panning");
  }

  function openViewer(diagram) {
    const cloned = cloneDiagramSvg(diagram);
    if (!cloned) {
      return;
    }

    ensureModal();
    closeViewer();

    active = cloned;
    state = { scale: 1 };
    viewport.replaceChildren(active.svg);
    modal.classList.add("is-open");
    document.body.classList.add("diagram-viewer-open");
    modal.querySelector("[data-diagram-viewer-close]").focus();

    window.requestAnimationFrame(() => {
      if (active) {
        resetDiagram();
      }
    });
  }

  function closeViewer() {
    if (!active) {
      return;
    }

    viewport.textContent = "";
    viewport.removeAttribute("style");
    modal.classList.remove("is-open");
    document.body.classList.remove("diagram-viewer-open");
    active = null;
    pan = null;
    stage.classList.remove("is-panning");
  }

  function prepareDiagram(diagram) {
    if (prepared.has(diagram) || diagram.closest(".diagram-viewer__modal") || !findRenderedSvg(diagram)) {
      return;
    }

    prepared.add(diagram);
    diagram.classList.add("diagram-viewer-inline__diagram");
    diagram.setAttribute("role", "button");
    diagram.setAttribute("tabindex", "0");
    diagram.setAttribute("aria-label", "Open diagram in viewer");
    diagram.addEventListener("click", () => openViewer(diagram));
    diagram.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") {
        return;
      }

      event.preventDefault();
      openViewer(diagram);
    });

    const wrapper = document.createElement("div");
    wrapper.className = "diagram-viewer-inline";

    const actions = document.createElement("div");
    actions.className = "diagram-viewer-inline__actions";

    const button = document.createElement("button");
    button.className = "diagram-viewer__open";
    button.type = "button";
    button.textContent = "Open diagram";
    button.setAttribute("aria-label", "Open diagram in viewer");
    button.addEventListener("click", () => openViewer(diagram));

    actions.appendChild(button);
    diagram.parentNode.insertBefore(wrapper, diagram);
    wrapper.appendChild(actions);
    wrapper.appendChild(diagram);
  }

  function scan() {
    document.querySelectorAll(".md-typeset .mermaid").forEach(prepareDiagram);
  }

  function scheduleScan() {
    window.setTimeout(scan, 0);
    window.setTimeout(scan, 250);
    window.setTimeout(scan, 1000);
  }

  if (window.document$) {
    window.document$.subscribe(() => {
      closeViewer();
      scheduleScan();
    });
  } else {
    document.addEventListener("DOMContentLoaded", scheduleScan);
  }

  new MutationObserver(scheduleScan).observe(document.documentElement, {
    childList: true,
    subtree: true,
  });

  scheduleScan();
})();
