/* lanevis dashboard — small multiples over mmengine scalar logs.
 *
 * One panel per metric, one colour per run. Colour is bound to the run's slot
 * from the full loaded list, so toggling runs never repaints the survivors.
 * Hovering any panel drives the crosshair in every panel, which is what makes a
 * loss spike legible against the validation curve.
 */
(function () {
  "use strict";

  var D = window.LANEVIS;
  var SVG_NS = "http://www.w3.org/2000/svg";

  var state = {
    visible: {},           // run name -> bool
    xAxis: "epoch",
    smoothing: 0.6,
    yScale: "linear",
    hoverX: null,
    hoverPanel: null,
    theme: null            // null = follow OS
  };

  D.runs.forEach(function (r) { state.visible[r.name] = true; });

  // ---------------------------------------------------------------- helpers
  function el(tag, attrs, text) {
    var node = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === "class") node.className = attrs[k];
      else node.setAttribute(k, attrs[k]);
    }
    if (text != null) node.textContent = text;
    return node;
  }

  function svgEl(tag, attrs) {
    var node = document.createElementNS(SVG_NS, tag);
    if (attrs) for (var k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  function isDark() {
    if (state.theme) return state.theme === "dark";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function runColor(run) {
    var table = isDark() ? D.palette.dark : D.palette.light;
    return table[run.slot % table.length];
  }

  function visibleRuns() {
    return D.runs.filter(function (r) { return state.visible[r.name]; });
  }

  /* Compact and readable: counts stay counts (25.6k, not 2.56e+4), rates keep
   * enough digits to separate runs, and only genuinely tiny values (a learning
   * rate) fall back to exponent form. */
  function fmt(v) {
    if (v == null || !isFinite(v)) return "—";
    var a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
    if (a >= 1e4) return (v / 1e3).toFixed(a >= 1e5 ? 0 : 1) + "k";
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(2);
    if (a >= 1) return v.toFixed(3);
    if (a >= 0.001) return v.toFixed(4);
    return v.toExponential(1);
  }

  /* Axis ticks are formatted against the tick *step*, not the magnitude: GPU
   * memory stepping 11742 -> 11744 needs different digits from a loss stepping
   * 0 -> 20, and formatting both by magnitude collapses the first to two
   * identical "11.7k" labels. */
  function fmtTick(v, step) {
    if (!isFinite(v)) return "";
    if (v === 0) return "0";
    var a = Math.abs(v);
    step = Math.abs(step) || a / 4;
    if (step >= 1e6) return (v / 1e6).toFixed(0) + "M";
    if (step >= 1e3 && a >= 1e4) return (v / 1e3).toFixed(step >= 1e4 ? 0 : 1) + "k";
    if (step < 1e-4) return v.toExponential(1);
    var dec = Math.max(0, Math.min(6, -Math.floor(Math.log10(step))));
    return v.toFixed(dec);
  }

  function fmtX(v) {
    if (state.xAxis === "epoch") return (Math.abs(v - Math.round(v)) < 0.05)
      ? String(Math.round(v)) : v.toFixed(1);
    return v >= 1000 ? (v / 1000).toFixed(1) + "k" : String(Math.round(v));
  }

  /* TensorBoard's debiased EMA. Applied to training curves only: validation is
   * already one point per interval and smoothing it would hide the thing being
   * measured. */
  function ema(values, weight) {
    if (!(weight > 0) || !values.length) return values;
    var w = Math.min(weight, 0.999), out = new Array(values.length);
    var last = 0, debias = 0;
    for (var i = 0; i < values.length; i++) {
      last = last * w + (1 - w) * values[i];
      debias = debias * w + (1 - w);
      out[i] = debias > 0 ? last / debias : values[i];
    }
    return out;
  }

  function seriesFor(run, key) {
    var byRun = D.series[run.name];
    if (!byRun) return null;
    var s = byRun[key];
    if (!s || !s.y.length) return null;
    var xs = s.x;
    if (state.xAxis === "iter") {
      var n = run.iters_per_epoch || 1;
      xs = s.x.map(function (v) { return v * n; });
    }
    var ys = s.val ? s.y : ema(s.y, state.smoothing);
    return { x: xs, y: ys, raw: s.y, val: !!s.val };
  }

  function niceTicks(lo, hi, count) {
    if (!(hi > lo)) return [lo];
    var span = hi - lo;
    var step = Math.pow(10, Math.floor(Math.log10(span / count)));
    var err = span / count / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var out = [], t = Math.ceil(lo / step) * step;
    for (; t <= hi + step * 1e-6; t += step) out.push(Math.abs(t) < step * 1e-6 ? 0 : t);
    return out;
  }

  function logTicks(lo, hi) {
    var out = [];
    for (var e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      [1, 2, 5].forEach(function (m) {
        var v = m * Math.pow(10, e);
        if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
      });
    }
    return out.length >= 2 ? out : niceTicks(lo, hi, 4);
  }

  // ------------------------------------------------------------- x domain
  function xDomain() {
    var lo = Infinity, hi = -Infinity;
    visibleRuns().forEach(function (run) {
      var byRun = D.series[run.name] || {};
      for (var key in byRun) {
        var s = seriesFor(run, key);
        if (!s) continue;
        if (s.x[0] < lo) lo = s.x[0];
        if (s.x[s.x.length - 1] > hi) hi = s.x[s.x.length - 1];
      }
    });
    if (!isFinite(lo)) return [0, 1];
    if (hi <= lo) hi = lo + 1;
    return [state.xAxis === "epoch" ? 0 : 0, hi];
  }

  // ----------------------------------------------------------------- panel
  function Panel(container, key, note) {
    this.key = key;
    this.card = el("div", { class: "card panelbox" });
    this.card.appendChild(el("h2", null, D.pretty[key] || key));
    if (note) this.card.appendChild(el("p", { class: "note" }, note));
    this.plot = el("div", { class: "plot" });
    this.card.appendChild(this.plot);
    this.tip = el("div", { class: "tip" });
    this.plot.appendChild(this.tip);
    this.svg = svgEl("svg", { preserveAspectRatio: "none" });
    this.plot.insertBefore(this.svg, this.tip);
    container.appendChild(this.card);

    var self = this;
    this.svg.addEventListener("mousemove", function (ev) { self.onMove(ev); });
    this.svg.addEventListener("mouseleave", function () {
      state.hoverX = null; state.hoverPanel = null; renderHover();
    });
    this.svg.addEventListener("touchmove", function (ev) {
      if (ev.touches.length) { self.onMove(ev.touches[0]); ev.preventDefault(); }
    }, { passive: false });
  }

  Panel.prototype.onMove = function (ev) {
    if (!this.geom) return;
    var rect = this.svg.getBoundingClientRect();
    var px = (ev.clientX - rect.left) / rect.width * this.geom.w;
    var g = this.geom;
    var frac = (px - g.left) / Math.max(1, g.w - g.left - g.right);
    state.hoverX = g.x0 + frac * (g.x1 - g.x0);
    state.hoverPanel = this.key;
    renderHover();
  };

  Panel.prototype.draw = function (xd) {
    var runs = visibleRuns();
    // End-of-line labels need their own gutter, or the card's overflow clips
    // them. Reserve it up front rather than letting the text run past the plot.
    var labelEnds = runs.length <= 4;
    var W = 520, H = 210, left = 52, right = labelEnds ? 54 : 14, top = 10, bottom = 26;
    var lines = [];
    var lo = Infinity, hi = -Infinity, anyVal = false;

    runs.forEach(function (run) {
      var s = seriesFor(run, this.key);
      if (!s) return;
      anyVal = anyVal || s.val;
      for (var i = 0; i < s.y.length; i++) {
        if (!isFinite(s.y[i])) continue;
        if (s.y[i] < lo) lo = s.y[i];
        if (s.y[i] > hi) hi = s.y[i];
      }
      lines.push({ run: run, s: s });
    }, this);

    this.svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    this.svg.setAttribute("style", "height:" + H + "px;max-height:" + H + "px");
    while (this.svg.firstChild) this.svg.removeChild(this.svg.firstChild);

    if (!lines.length) {
      this.svg.appendChild(svgEl("rect", { x: 0, y: 0, width: W, height: H, fill: "none" }));
      var none = svgEl("text", {
        x: W / 2, y: H / 2, "text-anchor": "middle", "font-size": 12,
        fill: "var(--text-muted)", "font-family": "var(--font)"
      });
      none.textContent = "no data in the selected runs";
      this.svg.appendChild(none);
      this.geom = null;
      return;
    }

    var useLog = state.yScale === "log" && this.logOk && lo > 0;
    /* A flat metric (constant GPU memory, a frozen lr, a loss term pinned at 0)
     * must render as a flat line. Testing hi === lo is not enough: the EMA
     * divides by a debias term, so a constant input comes back with ~1e-12
     * relative float noise, and an auto-scaled axis magnifies that into a
     * square wave with three identical tick labels. */
    var flatEps = Math.max(Math.abs(hi), Math.abs(lo)) * 1e-9;
    if (hi - lo <= flatEps) {
      var mid = (hi + lo) / 2;
      var halfSpan = Math.abs(mid) * 0.1 || 1;
      lo = mid - halfSpan;
      hi = mid + halfSpan;
    }
    var pad = (hi - lo) * 0.08;
    var yLo = useLog ? lo / 1.3 : lo - pad;
    var yHi = useLog ? hi * 1.3 : hi + pad;

    function sx(v) { return left + (v - xd[0]) / (xd[1] - xd[0] || 1) * (W - left - right); }
    function sy(v) {
      if (useLog) {
        var t = (Math.log10(v) - Math.log10(yLo)) / (Math.log10(yHi) - Math.log10(yLo) || 1);
        return H - bottom - t * (H - top - bottom);
      }
      return H - bottom - (v - yLo) / (yHi - yLo || 1) * (H - top - bottom);
    }

    var yTicks = useLog ? logTicks(yLo, yHi) : niceTicks(yLo, yHi, 4);
    var yStep = yTicks.length > 1 ? yTicks[1] - yTicks[0] : (yHi - yLo) / 4;
    yTicks.forEach(function (t) {
      var y = sy(t);
      if (y < top - 1 || y > H - bottom + 1) return;
      this.svg.appendChild(svgEl("line", {
        x1: left, x2: W - right, y1: y, y2: y,
        stroke: "var(--grid)", "stroke-width": 1, "shape-rendering": "crispEdges"
      }));
      var label = svgEl("text", {
        x: left - 7, y: y + 3.5, "text-anchor": "end", "font-size": 10,
        fill: "var(--text-muted)", "font-family": "var(--font)",
        "font-variant-numeric": "tabular-nums"
      });
      label.textContent = fmtTick(t, yStep);
      this.svg.appendChild(label);
    }, this);

    niceTicks(xd[0], xd[1], 5).forEach(function (t) {
      var x = sx(t);
      if (x < left - 1 || x > W - right + 1) return;
      var label = svgEl("text", {
        x: x, y: H - bottom + 14, "text-anchor": "middle", "font-size": 10,
        fill: "var(--text-muted)", "font-family": "var(--font)",
        "font-variant-numeric": "tabular-nums"
      });
      label.textContent = fmtX(t);
      this.svg.appendChild(label);
    }, this);

    this.svg.appendChild(svgEl("line", {
      x1: left, x2: W - right, y1: H - bottom, y2: H - bottom,
      stroke: "var(--axis)", "stroke-width": 1, "shape-rendering": "crispEdges"
    }));

    var axisTitle = svgEl("text", {
      x: W - right, y: H - 2, "text-anchor": "end", "font-size": 9.5,
      fill: "var(--text-muted)", "font-family": "var(--font)"
    });
    axisTitle.textContent = state.xAxis === "epoch" ? "epoch" : "iteration";
    this.svg.appendChild(axisTitle);

    lines.forEach(function (line) {
      var color = runColor(line.run), pts = [];
      for (var i = 0; i < line.s.y.length; i++) {
        if (!isFinite(line.s.y[i])) continue;
        if (useLog && line.s.y[i] <= 0) continue;
        pts.push(sx(line.s.x[i]).toFixed(2) + "," + sy(line.s.y[i]).toFixed(2));
      }
      if (!pts.length) return;
      this.svg.appendChild(svgEl("polyline", {
        points: pts.join(" "), fill: "none", stroke: color,
        "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round"
      }));
      // Validation series are sparse: mark the actual measurements.
      if (line.s.val && line.s.y.length <= 60) {
        for (var j = 0; j < line.s.y.length; j++) {
          if (!isFinite(line.s.y[j])) continue;
          if (useLog && line.s.y[j] <= 0) continue;
          this.svg.appendChild(svgEl("circle", {
            cx: sx(line.s.x[j]), cy: sy(line.s.y[j]), r: 4,
            fill: color, stroke: "var(--surface-1)", "stroke-width": 2
          }));
        }
      }
      if (labelEnds) {
        var k = line.s.y.length - 1;
        while (k > 0 && !isFinite(line.s.y[k])) k--;
        var tx = sx(line.s.x[k]), ty = sy(line.s.y[k]);
        var text = fmt(line.s.y[k]);
        var wide = text.length * 5.7;            // ~advance width at 10px
        var overflows = tx + 6 + wide > W - 2;
        var t = svgEl("text", {
          x: overflows ? W - 2 : tx + 6,
          y: Math.max(top + 4, Math.min(ty + 3.5, H - bottom - 1)),
          "text-anchor": overflows ? "end" : "start",
          "font-size": 10, fill: "var(--text-secondary)", "font-family": "var(--font)",
          "font-variant-numeric": "tabular-nums", "paint-order": "stroke",
          stroke: "var(--surface-1)", "stroke-width": 3, "stroke-linejoin": "round"
        });
        t.textContent = text;
        this.svg.appendChild(t);
      }
    }, this);

    this.hoverLayer = svgEl("g", null);
    this.svg.appendChild(this.hoverLayer);

    this.geom = { w: W, h: H, left: left, right: right, top: top, bottom: bottom,
                  x0: xd[0], x1: xd[1], sx: sx, sy: sy, lines: lines, useLog: useLog };
  };

  Panel.prototype.drawHover = function () {
    if (!this.hoverLayer) return;
    while (this.hoverLayer.firstChild) this.hoverLayer.removeChild(this.hoverLayer.firstChild);
    this.tip.className = "tip";
    var g = this.geom;
    if (!g || state.hoverX == null) return;

    var hits = [];
    g.lines.forEach(function (line) {
      var best = -1, bestD = Infinity;
      for (var i = 0; i < line.s.x.length; i++) {
        var d = Math.abs(line.s.x[i] - state.hoverX);
        if (d < bestD) { bestD = d; best = i; }
      }
      if (best < 0 || !isFinite(line.s.y[best])) return;
      hits.push({ run: line.run, x: line.s.x[best], y: line.s.y[best], raw: line.s.raw[best] });
    });
    if (!hits.length) return;

    var hx = g.sx(state.hoverX);
    if (hx < g.left || hx > g.w - g.right) return;

    this.hoverLayer.appendChild(svgEl("line", {
      x1: hx, x2: hx, y1: g.top, y2: g.h - g.bottom,
      stroke: "var(--axis)", "stroke-width": 1, "shape-rendering": "crispEdges"
    }));
    hits.forEach(function (hit) {
      if (g.useLog && hit.y <= 0) return;
      this.hoverLayer.appendChild(svgEl("circle", {
        cx: g.sx(hit.x), cy: g.sy(hit.y), r: 4.5,
        fill: runColor(hit.run), stroke: "var(--surface-1)", "stroke-width": 2
      }));
    }, this);

    if (state.hoverPanel !== this.key) return;

    var head = state.xAxis === "epoch"
      ? "epoch " + hits[0].x.toFixed(2)
      : "iter " + Math.round(hits[0].x);
    var html = '<div class="head">' + head + "</div><table>";
    hits.sort(function (a, b) { return b.y - a.y; }).forEach(function (hit) {
      var smoothed = !hit.raw || Math.abs(hit.raw - hit.y) < 1e-12 ? "" :
        ' <span style="color:var(--text-muted)">(' + fmt(hit.raw) + ")</span>";
      html += '<tr><td class="k"><span class="dot" style="background:' + runColor(hit.run) +
        '"></span>' + escapeHtml(hit.run.label) + '</td><td class="v">' + fmt(hit.y) + smoothed + "</td></tr>";
    });
    html += "</table>";
    this.tip.innerHTML = html;
    this.tip.className = "tip on";
    // Park the tooltip in the corner away from the crosshair, so it never
    // covers the part of the curve the reader is pointing at.
    var boxW = this.tip.offsetWidth || 170;
    var plotW = this.plot.clientWidth || 1;
    var px = hx / g.w * plotW;
    this.tip.style.left = (px > plotW / 2 ? 4 : Math.max(4, plotW - boxW - 4)) + "px";
    this.tip.style.top = "0px";
  };

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // ----------------------------------------------------------------- build
  var panels = [];

  function buildControls(root) {
    var box = el("div", { class: "controls panelbox" });

    var row1 = el("div", { class: "controls-row" });
    var chips = el("div", { class: "runs" });
    D.runs.forEach(function (run) {
      var chip = el("button", { class: "run-chip", type: "button", "aria-pressed": "true",
                                title: run.name + "\n" + run.path });
      var swatch = el("span", { class: "swatch" });
      swatch.style.background = runColor(run);
      chip.appendChild(swatch);
      chip.appendChild(el("span", { class: "name" }, run.label));
      var conf = run.conf == null
        ? el("span", { class: "conf unknown", title: "confidence threshold not found in the run config" }, "conf ?")
        : el("span", { class: "conf" }, "conf " + run.conf.toFixed(2));
      chip.appendChild(conf);
      chip.addEventListener("click", function () {
        state.visible[run.name] = !state.visible[run.name];
        chip.setAttribute("aria-pressed", state.visible[run.name] ? "true" : "false");
        renderAll();
      });
      run._chip = chip;
      run._swatch = swatch;
      chips.appendChild(chip);
    });
    row1.appendChild(chips);
    box.appendChild(row1);

    var row2 = el("div", { class: "controls-row" });

    row2.appendChild(toggleField("x axis", [["epoch", "epoch"], ["iter", "iteration"]],
      function () { return state.xAxis; },
      function (v) { state.xAxis = v; renderAll(); }));

    row2.appendChild(toggleField("y scale", [["linear", "linear"], ["log", "log (losses)"]],
      function () { return state.yScale; },
      function (v) { state.yScale = v; renderAll(); }));

    var sf = el("div", { class: "field" });
    sf.appendChild(el("label", { for: "smooth" }, "smoothing (training)"));
    var slider = el("input", { type: "range", id: "smooth", min: "0", max: "0.98", step: "0.02" });
    slider.value = String(state.smoothing);
    var readout = el("span", { class: "value" }, state.smoothing.toFixed(2));
    slider.addEventListener("input", function () {
      state.smoothing = parseFloat(slider.value);
      readout.textContent = state.smoothing.toFixed(2);
      renderAll();
    });
    sf.appendChild(slider);
    sf.appendChild(readout);
    row2.appendChild(sf);

    var all = el("button", { type: "button" }, "all runs");
    all.addEventListener("click", function () {
      D.runs.forEach(function (r) { state.visible[r.name] = true; r._chip.setAttribute("aria-pressed", "true"); });
      renderAll();
    });
    var none = el("button", { type: "button" }, "none");
    none.addEventListener("click", function () {
      D.runs.forEach(function (r) { state.visible[r.name] = false; r._chip.setAttribute("aria-pressed", "false"); });
      renderAll();
    });
    row2.appendChild(all);
    row2.appendChild(none);
    box.appendChild(row2);

    root.appendChild(box);
  }

  function toggleField(labelText, options, get, set) {
    var field = el("div", { class: "field" });
    field.appendChild(el("label", null, labelText));
    options.forEach(function (opt) {
      var button = el("button", { type: "button", "aria-pressed": get() === opt[0] ? "true" : "false" }, opt[1]);
      button.addEventListener("click", function () {
        set(opt[0]);
        field.querySelectorAll("button").forEach(function (b, i) {
          b.setAttribute("aria-pressed", options[i][0] === get() ? "true" : "false");
        });
      });
      field.appendChild(button);
    });
    return field;
  }

  function buildPanels(root) {
    D.panels.forEach(function (group) {
      root.appendChild(el("div", { class: "section-title" }, group.title));
      var grid = el("div", { class: "grid" });
      group.keys.forEach(function (key) {
        var panel = new Panel(grid, key, group.note || "");
        panel.logOk = !!group.log_default;
        panels.push(panel);
      });
      root.appendChild(grid);
    });
  }

  function buildTable(root) {
    root.appendChild(el("div", { class: "section-title" },
      "Best " + D.primary + " and when the curve got there"));
    var card = el("div", { class: "tablecard panelbox" });
    var table = el("table", { class: "data" });
    var head = el("tr");
    ["run", "conf", "epochs", "best", "@epoch", "within 0.2 pp", "within 0.5 pp", "final", "final − best"]
      .forEach(function (h) { head.appendChild(el("th", null, h)); });
    table.appendChild(el("thead", null)).appendChild(head);
    var body = el("tbody");
    table.appendChild(body);
    card.appendChild(table);
    card.appendChild(el("p", { class: "caption" },
      "“within x pp” is the earliest epoch whose running best is that close to the run's own best — " +
      "the first epoch you could have stopped at. It is not the argmax."));
    root.appendChild(card);
    return body;
  }

  var tableBody;

  function renderTable() {
    while (tableBody.firstChild) tableBody.removeChild(tableBody.firstChild);
    D.summary.forEach(function (s) {
      var run = D.runs.filter(function (r) { return r.name === s.run; })[0];
      var tr = el("tr", state.visible[s.run] ? {} : { class: "dim" });
      var td = el("td", { class: "run" });
      var swatch = el("span", { class: "swatch" });
      swatch.style.background = run ? runColor(run) : "transparent";
      td.appendChild(swatch);
      td.appendChild(el("span", null, s.label));
      tr.appendChild(td);
      [
        s.conf == null ? "—" : s.conf.toFixed(2),
        s.epochs || "—",
        s.best == null ? "—" : (100 * s.best).toFixed(2),
        s.best_epoch == null ? "—" : String(Math.round(s.best_epoch)),
        s.plateau_002 == null ? "—" : String(Math.round(s.plateau_002)),
        s.plateau_005 == null ? "—" : String(Math.round(s.plateau_005)),
        s.final == null ? "—" : (100 * s.final).toFixed(2),
        s.drift == null ? "—" : ((s.drift >= 0 ? "+" : "") + (100 * s.drift).toFixed(2))
      ].forEach(function (v) { tr.appendChild(el("td", null, String(v))); });
      tableBody.appendChild(tr);
    });
  }

  function renderHover() {
    panels.forEach(function (p) { p.drawHover(); });
  }

  function renderAll() {
    var xd = xDomain();
    D.runs.forEach(function (r) { if (r._swatch) r._swatch.style.background = runColor(r); });
    panels.forEach(function (p) { p.draw(xd); });
    renderTable();
    renderHover();
  }

  // ------------------------------------------------------------------ init
  function main() {
    var wrap = el("div", { class: "wrap" });

    var header = el("header");
    header.appendChild(el("h1", null, D.title || "Training curves"));
    header.appendChild(el("span", { class: "sub" },
      D.runs.length + (D.runs.length === 1 ? " run" : " runs") + " · " + D.generated));
    header.appendChild(el("div", { class: "spacer" }));
    var themeBtn = el("button", { type: "button", title: "light / dark / follow system" }, "theme");
    themeBtn.addEventListener("click", function () {
      state.theme = state.theme === "dark" ? "light" : (state.theme === "light" ? null : "dark");
      if (state.theme) document.documentElement.setAttribute("data-theme", state.theme);
      else document.documentElement.removeAttribute("data-theme");
      themeBtn.textContent = state.theme ? "theme: " + state.theme : "theme";
      renderAll();
    });
    header.appendChild(themeBtn);
    wrap.appendChild(header);

    if (!D.runs.length) {
      wrap.appendChild(el("div", { class: "empty panelbox" }, "No runs were found."));
      document.body.appendChild(wrap);
      return;
    }

    if (D.warnings && D.warnings.length) {
      var warn = el("div", { class: "controls panelbox" });
      D.warnings.forEach(function (w) { warn.appendChild(el("p", { class: "note" }, w)); });
      wrap.appendChild(warn);
    }

    buildControls(wrap);
    buildPanels(wrap);
    tableBody = buildTable(wrap);
    document.body.appendChild(wrap);

    renderAll();
    window.addEventListener("resize", function () { renderHover(); });
    if (window.matchMedia) {
      var mq = window.matchMedia("(prefers-color-scheme: dark)");
      if (mq.addEventListener) mq.addEventListener("change", renderAll);
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", main);
  else main();
})();
