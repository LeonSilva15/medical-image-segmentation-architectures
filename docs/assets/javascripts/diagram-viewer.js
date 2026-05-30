(function () {
  const prepared = new WeakSet();
  let active = null;
  let modal = null;
  let viewport = null;
  let stage = null;
  let scaleLabel = null;
  let state = { scale: 1, x: 0, y: 0 };
  let pan = null;

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
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
    modal.querySelector("[data-diagram-viewer-reset]").addEventListener("click", fitDiagram);
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

  function updateTransform() {
    viewport.style.setProperty("--diagram-viewer-scale", state.scale.toString());
    viewport.style.setProperty("--diagram-viewer-x", `${state.x}px`);
    viewport.style.setProperty("--diagram-viewer-y", `${state.y}px`);
    scaleLabel.textContent = `${Math.round(state.scale * 100)}%`;
  }

  function fitDiagram() {
    if (!active) {
      return;
    }

    state = { scale: 1, x: 0, y: 0 };
    updateTransform();

    window.requestAnimationFrame(() => {
      const diagramRect = active.diagram.getBoundingClientRect();
      const stageRect = stage.getBoundingClientRect();
      const widthRatio = (stageRect.width * 0.92) / diagramRect.width;
      const heightRatio = (stageRect.height * 0.88) / diagramRect.height;
      state.scale = clamp(Math.min(widthRatio, heightRatio, 1), 0.15, 4);
      state.x = 0;
      state.y = 0;
      updateTransform();
    });
  }

  function zoomBy(factor) {
    state.scale = clamp(state.scale * factor, 0.15, 8);
    updateTransform();
  }

  function onWheel(event) {
    if (!active) {
      return;
    }

    event.preventDefault();
    zoomBy(event.deltaY < 0 ? 1.12 : 1 / 1.12);
  }

  function onPointerDown(event) {
    if (!active || event.button !== 0) {
      return;
    }

    pan = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      x: state.x,
      y: state.y,
    };
    stage.classList.add("is-panning");
    stage.setPointerCapture(event.pointerId);
  }

  function onPointerMove(event) {
    if (!pan || event.pointerId !== pan.pointerId) {
      return;
    }

    state.x = pan.x + event.clientX - pan.startX;
    state.y = pan.y + event.clientY - pan.startY;
    updateTransform();
  }

  function endPan(event) {
    if (!pan || event.pointerId !== pan.pointerId) {
      return;
    }

    pan = null;
    stage.classList.remove("is-panning");
  }

  function openViewer(diagram) {
    ensureModal();

    const placeholder = document.createElement("div");
    placeholder.hidden = true;
    diagram.parentNode.insertBefore(placeholder, diagram);

    active = {
      diagram,
      placeholder,
      parent: placeholder.parentNode,
    };

    viewport.appendChild(diagram);
    modal.classList.add("is-open");
    document.body.classList.add("diagram-viewer-open");
    modal.querySelector("[data-diagram-viewer-close]").focus();
    fitDiagram();
  }

  function closeViewer() {
    if (!active) {
      return;
    }

    active.parent.insertBefore(active.diagram, active.placeholder);
    active.placeholder.remove();
    viewport.textContent = "";
    modal.classList.remove("is-open");
    document.body.classList.remove("diagram-viewer-open");
    active = null;
    pan = null;
    stage.classList.remove("is-panning");
  }

  function prepareDiagram(diagram) {
    if (prepared.has(diagram) || diagram.closest(".diagram-viewer__modal")) {
      return;
    }

    prepared.add(diagram);

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
    document.querySelectorAll(".md-typeset div.mermaid").forEach(prepareDiagram);
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
})();
