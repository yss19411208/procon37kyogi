"use strict";

const TEAM_COLORS = [
  "#e4572e", "#17bebb", "#f5c518", "#9b7ede", "#5ec8f0",
  "#6ee7a8", "#f472b6", "#fb923c", "#c084fc", "#94a3b8",
];
const TERRAIN_COLORS = ["#d8c9a3", null, "#9c7a55", "#2e5d74"]; // 1(道路)は状態で塗る
const ROAD_COLORS = ["#6f97a8", "#c98a3c", "#b5453a"];
const BRAND_COLORS = [
  "#ff7f50", "#5ec8f0", "#f7d154", "#a78bfa", "#6ee7a8",
  "#f472b6", "#fb923c", "#38bdf8", "#c4b5fd", "#facc15",
];

const state = {
  data: null,
  day: 0,
  step: 0,
  visible: [true, true, true, true],
  playing: false,
  timer: null,
  selectedCell: null,
  centers: [],
  size: 10,
  focus: 0,
  speed: 1,
  // **止まっている車を光らせるか**（キーボードの f で切り替え）。
  showStill: false,
  // 1台だけ見たいときの席番号（キーボードの1〜9）。
  stillOnly: null,
};

const BASE_INTERVAL_MS = 140;

const el = (id) => document.getElementById(id);
const canvas = el("map");
const ctx = canvas.getContext("2d");

// ---------- データ読み込み ----------

function loadDataset(slug) {
  const cached = window.HEXA_DATASETS && window.HEXA_DATASETS[slug];
  if (cached) {
    applyDataset(cached);
    return;
  }
  const script = document.createElement("script");
  script.src = "data/" + slug + ".js" + (window.HEXA_CACHE_BUSTER || "");
  script.onload = () => {
    const payload = window.HEXA_DATASETS && window.HEXA_DATASETS[slug];
    if (payload) applyDataset(payload);
    else showEmpty("データを読み込めませんでした: " + slug);
  };
  script.onerror = () => showEmpty("データファイルが見つかりません: data/" + slug + ".js");
  document.head.appendChild(script);
}

function applyDataset(payload) {
  state.data = payload;
  state.day = 0;
  state.step = 0;
  state.visible = payload.teams.map(() => true);
  state.selectedCell = null;
  state.focus = 0;
  buildDaySelect();
  buildTeamList();
  buildFocusSelect();
  layout();
  syncStepRange();
  render();
}

function showEmpty(message) {
  el("head").textContent = message;
}

// ---------- UI 構築 ----------

function buildDatasetSelect() {
  const select = el("dataset");
  const index = window.HEXA_INDEX || [];
  select.innerHTML = "";
  if (index.length === 0) {
    showEmpty("解答データがまだありません。main.py の「問題を解く」を実行してください。");
    return;
  }
  index.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.slug;
    option.textContent = entry.title + "（" + entry.createdAt + "）";
    select.appendChild(option);
  });
  select.onchange = () => loadDataset(select.value);
  loadDataset(index[0].slug);
}

function buildDaySelect() {
  const select = el("day");
  select.innerHTML = "";
  const dayCount = state.data.problem.daySteps.length;
  for (let day = 0; day < dayCount; day += 1) {
    const option = document.createElement("option");
    option.value = String(day);
    option.textContent = (day + 1) + "日目（" + state.data.problem.daySteps[day] + "ステップ）";
    select.appendChild(option);
  }
  select.onchange = () => {
    state.day = Number(select.value);
    state.step = 0;
    syncStepRange();
    render();
  };
}

function buildTeamList() {
  const container = el("teams");
  container.innerHTML = "";
  state.data.teams.forEach((team, index) => {
    const button = document.createElement("button");
    button.className = "team-row";
    button.style.color = TEAM_COLORS[index % TEAM_COLORS.length];
    button.setAttribute("aria-pressed", "true");
    button.innerHTML =
      '<span class="swatch"></span>' +
      '<span class="name">' + team.name + "（" + team.rank + "位）</span>" +
      '<span class="score">種' + team.score.brand +
      " 日" + team.score.dailyBrand +
      " 玉" + team.score.noodles + "</span>";
    button.onclick = () => {
      state.visible[index] = !state.visible[index];
      button.setAttribute("aria-pressed", String(state.visible[index]));
      button.style.opacity = state.visible[index] ? "1" : "0.4";
      render();
    };
    container.appendChild(button);
  });
}

function buildFocusSelect() {
  const select = el("focus");
  select.innerHTML = "";
  state.data.teams.forEach((team, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = team.name;
    select.appendChild(option);
  });
  select.value = String(state.focus);
  select.onchange = () => {
    state.focus = Number(select.value);
    render();
  };
}

// ---------- 幾何 ----------

function layout() {
  const problem = state.data.problem;
  const wrap = canvas.parentElement;
  const ratio = window.devicePixelRatio || 1;
  const cssWidth = wrap.clientWidth;
  const cssHeight = wrap.clientHeight;
  canvas.width = cssWidth * ratio;
  canvas.height = cssHeight * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

  // 尖った頭の六角形。偶数行を右に半セルずらす（質問回答集その1 Q1）。
  const sizeByWidth = cssWidth / (Math.sqrt(3) * (problem.width + 0.5) + 1);
  const sizeByHeight = cssHeight / (1.5 * problem.height + 0.5 + 1);
  const size = Math.max(4, Math.min(sizeByWidth, sizeByHeight));
  state.size = size;

  const hexWidth = Math.sqrt(3) * size;
  const totalWidth = hexWidth * (problem.width + 0.5);
  const totalHeight = 1.5 * size * (problem.height - 1) + 2 * size;
  const offsetX = (cssWidth - totalWidth) / 2 + hexWidth / 2;
  const offsetY = (cssHeight - totalHeight) / 2 + size;

  state.centers = [];
  for (let row = 0; row < problem.height; row += 1) {
    for (let column = 0; column < problem.width; column += 1) {
      const shift = row % 2 === 0 ? hexWidth / 2 : 0;
      state.centers.push({
        x: offsetX + column * hexWidth + shift,
        y: offsetY + row * 1.5 * size,
      });
    }
  }
}

function hexPath(cx, cy, size) {
  ctx.beginPath();
  for (let corner = 0; corner < 6; corner += 1) {
    const angle = (Math.PI / 180) * (60 * corner - 90);
    const x = cx + size * Math.cos(angle);
    const y = cy + size * Math.sin(angle);
    if (corner === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
}

// ---------- 描画 ----------

function currentDay() {
  return state.data.teams.map((team) => team.days[state.day]);
}

function maxStep() {
  const days = currentDay();
  return Math.max(0, Math.min.apply(null, days.map((d) => d.steps.length - 1)));
}

function syncStepRange() {
  const slider = el("step");
  slider.max = String(maxStep());
  slider.value = String(state.step);
}

function remainingStocks() {
  // 在庫はチームごとに独立で、日の開始時に最大在庫まで戻る（募集要項 p.10）。
  // なので「最大在庫 − そのチームがその日そのステップまでに取った数」で出せる。
  const problem = state.data.problem;
  const team = state.data.teams[state.focus];
  const stocks = new Map();
  problem.spots.forEach((spot) => stocks.set(spot.pos, spot.stocks));
  if (!team) return stocks;
  const day = team.days[state.day];
  day.acquisitions.forEach((event) => {
    if (event.step > state.step) return;
    stocks.set(event.pos, (stocks.get(event.pos) || 0) - 1);
  });
  return stocks;
}

function renderAgentPanel() {
  const container = el("agents");
  const team = state.data.teams[state.focus];
  container.innerHTML = "";
  if (!team) return;
  const day = team.days[state.day];
  const frame = day.steps[Math.min(state.step, day.steps.length - 1)];
  if (!frame) return;
  const limit = state.data.problem.fuelLimit;
  const justRefuelled = new Set(
    day.refuels.filter((e) => e.step === state.step).map((e) => e.agent)
  );
  const justAcquired = new Set(
    day.acquisitions.filter((e) => e.step === state.step).map((e) => e.agent)
  );

  frame.positions.forEach((position, index) => {
    const supply = team.kinds[index] === 1;
    const fuel = frame.fuels[index];
    const ratio = supply ? 1 : Math.max(0, Math.min(1, fuel / limit));
    const row = document.createElement("div");
    row.className = "agent-row";
    let mark = "";
    if (justAcquired.has(index)) mark = " ● 獲得";
    else if (justRefuelled.has(index)) mark = " ◆ 補給";
    const fuelText = supply ? "燃料 無制限" : `燃料 ${fuel} / ${limit}`;
    const level = ratio < 0.2 ? "low" : ratio < 0.5 ? "mid" : "";
    row.innerHTML =
      '<span class="who">' + (supply ? "補給車" : "巡回車") + (index + 1) + mark + "</span>" +
      '<span class="fuel-text">セル' + position + "</span>" +
      '<span class="fuel-text" style="grid-column:1/-1">' + fuelText + "</span>" +
      '<span class="gauge"><span class="' + level +
      '" style="width:' + (ratio * 100).toFixed(1) + '%"></span></span>';
    container.appendChild(row);
  });
}

function renderSpotPanel(stocks) {
  const container = el("spots");
  const problem = state.data.problem;
  container.innerHTML = "";
  const spots = problem.spots.slice().sort((a, b) => a.brand - b.brand || a.pos - b.pos);
  spots.forEach((spot) => {
    const left = Math.max(0, stocks.get(spot.pos) || 0);
    const row = document.createElement("div");
    row.className = "spot-row" + (left === 0 ? " empty" : "");
    row.innerHTML =
      '<i style="background:' + BRAND_COLORS[spot.brand % BRAND_COLORS.length] + '"></i>' +
      '<span class="left">系列' + spot.brand + " ・ セル" + spot.pos + "</span>" +
      "<span>" + left + " / " + spot.stocks + "</span>";
    container.appendChild(row);
  });
}

function render() {
  if (!state.data) return;
  const problem = state.data.problem;
  const days = currentDay();
  const roadStatus = new Map();
  days[0].roadStatus.forEach((entry) => roadStatus.set(entry.pos, entry.status));
  const spotByPos = new Map();
  problem.spots.forEach((spot) => spotByPos.set(spot.pos, spot));

  const stocks = remainingStocks();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let position = 0; position < problem.cells.length; position += 1) {
    const center = state.centers[position];
    if (!center) continue;
    const terrain = problem.cells[position];
    let fill = TERRAIN_COLORS[terrain];
    if (terrain === 1) fill = ROAD_COLORS[roadStatus.get(position) || 0];
    hexPath(center.x, center.y, state.size * 0.96);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = position === state.selectedCell ? "#ffffff" : "rgba(0,0,0,0.25)";
    ctx.lineWidth = position === state.selectedCell ? 2.5 : 1;
    ctx.stroke();

    const spot = spotByPos.get(position);
    if (spot) {
      const left = Math.max(0, stocks.get(position) || 0);
      ctx.beginPath();
      ctx.arc(center.x, center.y, state.size * 0.45, 0, Math.PI * 2);
      ctx.fillStyle = BRAND_COLORS[spot.brand % BRAND_COLORS.length];
      ctx.globalAlpha = left === 0 ? 0.28 : 1;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#12171b";
      ctx.stroke();
      if (state.size >= 7) {
        // 白い縁取りを先に描いてから数字を重ねる。系列の色が濃くても読める。
        ctx.font = "700 " + Math.round(state.size * 0.66) + "px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.lineWidth = Math.max(2, state.size * 0.16);
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.strokeText(String(left), center.x, center.y + state.size * 0.02);
        ctx.fillStyle = "#12171b";
        ctx.fillText(String(left), center.x, center.y + state.size * 0.02);
      }
    }
  }

  state.data.teams.forEach((team, teamIndex) => {
    if (!state.visible[teamIndex]) return;
    const day = team.days[state.day];
    const frame = day.steps[Math.min(state.step, day.steps.length - 1)];
    if (!frame) return;
    const color = TEAM_COLORS[teamIndex % TEAM_COLORS.length];
    const seen = new Map();
    frame.positions.forEach((position, agentIndex) => {
      const center = state.centers[position];
      if (!center) return;
      const stackIndex = seen.get(position) || 0;
      seen.set(position, stackIndex + 1);
      const spread = state.size * 0.34;
      const teamCount = Math.max(1, state.data.teams.length);
      const angle =
        (teamIndex * (360 / teamCount) + stackIndex * 29) * (Math.PI / 180);
      const x = center.x + Math.cos(angle) * spread;
      const y = center.y + Math.sin(angle) * spread;
      const radius = Math.max(3, state.size * 0.26);
      ctx.beginPath();
      if (team.kinds[agentIndex] === 1) {
        ctx.moveTo(x, y - radius);
        ctx.lineTo(x + radius, y);
        ctx.lineTo(x, y + radius);
        ctx.lineTo(x - radius, y);
        ctx.closePath();
      } else {
        ctx.arc(x, y, radius, 0, Math.PI * 2);
      }
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#0d1114";
      ctx.stroke();
    });
  });

  updateHead();
  updateReadout();
  renderAgentPanel();
  renderSpotPanel(stocks);
  renderDayTable();
}

// **試合ぜんぶの日別一覧。**
//
// 1日ずつ見ていくと「どの日が悪かったのか」が分かりません。日ごとの
// 系列・玉・移動回数・止まった割合・配った燃料を並べ、いちばん悪い日に
// 飛べるようにします。悪い日は**その前の晩に燃料を使い切った**ことが
// 原因なので、前日と並べて見られることが大事です。
function dayStats(team, dayIndex) {
  const day = team.days[dayIndex];
  const roads = new Map();
  (day.roadStatus || []).forEach((entry) => roads.set(entry.pos, entry.status));
  const cells = state.data.problem.cells;
  const last = day.steps.length - 1;
  let moved = 0;
  let still = 0;
  let moves = 0;
  let used = 0;
  for (let seat = 0; seat < team.kinds.length; seat += 1) {
    if (team.kinds[seat] === 1) continue;
    const busy = new Array(last).fill(false);
    for (let t = 0; t + 1 <= last; t += 1) {
      const from = day.steps[t].positions[seat];
      if (from === day.steps[t + 1].positions[seat]) continue;
      const cost = moveSteps(cells[from], roads.get(from) || 0);
      moves += 1;
      used += cost;
      for (let back = Math.max(0, t + 1 - cost); back <= t; back += 1) {
        if (back < last) busy[back] = true;
      }
    }
    for (let t = 0; t < last; t += 1) {
      if (busy[t]) moved += 1;
      else still += 1;
    }
  }
  const brands = new Set(day.acquisitions.map((e) => e.brand));
  let given = 0;
  (day.refuels || []).forEach((e) => {
    // 古いデータには入った量が入っていないので、そのときは回数だけ。
    if (typeof e.after === "number" && typeof e.before === "number"
        && e.after >= 0 && e.before >= 0) {
      given += e.after - e.before;
    } else {
      given = -1;
    }
  });
  if (given < 0) given = 0;
  return {
    brands: brands.size,
    balls: day.acquisitions.length,
    moves: moves,
    perMove: moves ? used / moves : 0,
    stillRate: moved + still ? still / (moved + still) : 0,
    given: given,
    startFuel: day.steps.length
      ? day.steps[0].fuels.reduce((a, b, i) =>
          team.kinds[i] === 1 ? a : a + b, 0)
      : 0,
  };
}

function renderDayTable() {
  const box = el("daytable");
  if (!box) return;
  const team = state.data.teams[state.focus];
  if (!team) { box.innerHTML = ""; return; }
  const rows = team.days.map((_day, index) => dayStats(team, index));
  let worst = 0;
  rows.forEach((row, index) => {
    if (row.stillRate > rows[worst].stillRate) worst = index;
  });
  let html = "<table class=\"daytable\"><tr>" +
    "<th>日</th><th>系列</th><th>玉</th><th>移動</th>" +
    "<th>1移動の歩</th><th>止まった</th><th>朝の燃料</th><th>配った</th></tr>";
  rows.forEach((row, index) => {
    const mark = index === worst ? " class=\"worst\"" : "";
    const here = index === state.day ? " style=\"outline:1px solid #fff\"" : "";
    html += "<tr" + mark + here + " data-day=\"" + index + "\">" +
      "<td>" + (index + 1) + "</td>" +
      "<td>" + row.brands + "</td>" +
      "<td>" + row.balls + "</td>" +
      "<td>" + row.moves + "</td>" +
      "<td>" + row.perMove.toFixed(2) + "</td>" +
      "<td>" + Math.round(row.stillRate * 100) + "%</td>" +
      "<td>" + row.startFuel + "</td>" +
      "<td>" + (row.given > 0 ? row.given : "-") + "</td>" +
      "</tr>";
  });
  html += "</table>";
  box.innerHTML = html;
  box.querySelectorAll("tr[data-day]").forEach((row) => {
    row.onclick = () => {
      state.day = Number(row.dataset.day);
      state.step = 0;
      buildDaySelect();
      syncStepRange();
      render();
    };
  });
}

function updateHead() {
  const data = state.data;
  const overlap = (data.overlapRate * 100).toFixed(1);
  el("head").innerHTML =
    "<strong>" + data.title + "</strong>" +
    "<span>マップ " + data.problem.height + "×" + data.problem.width +
    " / スポット " + data.problem.spots.length +
    " / 燃料上限 " + data.problem.fuelLimit +
    " / players " + data.problem.players + "</span>" +
    "<span>" + (state.day + 1) + "日目のステップ数 " +
    data.problem.daySteps[state.day] + "</span>" +
    "<span>混雑基準 " + data.problem.busyThreshold +
    " ・ 渋滞基準 " + data.problem.jammedThreshold + "</span>" +
    "<span>4体の位置一致率 " + overlap + "%</span>";
}

function updateReadout() {
  const days = currentDay();
  const total = maxStep();
  let acquisitions = 0;
  let refuels = 0;
  state.data.teams.forEach((team, index) => {
    if (!state.visible[index]) return;
    const day = team.days[state.day];
    acquisitions += day.acquisitions.filter((e) => e.step <= state.step).length;
    refuels += day.refuels.filter((e) => e.step <= state.step).length;
  });
  // **止まっている歩**を数えます。玉数の伸びないいちばんの原因が
  // これなので、振り返りのときに真っ先に見たい数字です。
  //
  // **「位置が同じ歩」をそのまま数えてはいけません。** 1回の移動は
  // 地形と渋滞で1〜4歩かかるので、移動の途中も位置は変わりません。
  // 位置が変わったところを拾い、その移動にかかる歩数（**移動元の
  // マス**で決まります）をさかのぼって「動いた」に付け替えます。
  //
  // 準決勝32x32を本物の渋滞で数えると、1日の半分近くが「燃料は
  // あるのに止まっている」でした（燃料切れは0歩）。理由は
  // 「どの店にも今日は着けない」で、そこを動きに変えるのがいまの
  // いちばんの伸びしろです（docs/POLISH.md v1.52.0）。
  const team = state.data.teams[state.focus];
  let still = 0;
  let dry = 0;
  let moved = 0;
  if (team) {
    const day = team.days[state.day];
    const roads = new Map();
    (day.roadStatus || []).forEach((entry) => roads.set(entry.pos, entry.status));
    const cells = state.data.problem.cells;
    const last = Math.min(state.step, day.steps.length - 1);
    for (let seat = 0; seat < team.kinds.length; seat += 1) {
      if (team.kinds[seat] === 1) continue;
      const busy = new Array(last).fill(false);
      for (let t = 0; t + 1 <= last; t += 1) {
        const from = day.steps[t].positions[seat];
        if (from === day.steps[t + 1].positions[seat]) continue;
        const cost = moveSteps(cells[from], roads.get(from) || 0);
        for (let back = Math.max(0, t + 1 - cost); back <= t; back += 1) {
          if (back < last) busy[back] = true;
        }
      }
      for (let t = 0; t < last; t += 1) {
        if (busy[t]) { moved += 1; continue; }
        still += 1;
        if (day.steps[t].fuels[seat] <= 0) dry += 1;
      }
    }
  }
  const all = Math.max(1, moved + still);
  el("readout").textContent =
    "ステップ " + state.step + " / " + total +
    "　表示中のAIの獲得 " + acquisitions + "玉　補給 " + refuels + "回" +
    "　／ 巡回車が動いた " + moved +
    "（" + Math.round((100 * moved) / all) + "%）" +
    " 止まった " + still +
    "（" + Math.round((100 * still) / all) + "% ／ うち燃料ぎれ " + dry + "）";
}

// **1回の移動にかかる歩数。** 移動元のマスで決まります（要項1.1）。
//   平地2 / 山地3 / 道路は順調1・混雑2・渋滞4
function moveSteps(terrain, status) {
  if (terrain === 1) return status === 2 ? 4 : status === 1 ? 2 : 1;
  if (terrain === 2) return 3;
  return 2;
}

// ---------- セルの詳細 ----------

function describeCell(position) {
  const problem = state.data.problem;
  const names = ["平地", "道路", "山地", "池"];
  const terrain = problem.cells[position];
  const row = Math.floor(position / problem.width);
  const column = position % problem.width;
  const lines = [
    "<b>セル " + position + "</b>（" + row + "行 " + column + "列）",
    "地形: " + names[terrain],
  ];

  if (terrain === 1) {
    const day = state.data.teams[0].days[state.day];
    const entry = day.roadStatus.find((item) => item.pos === position);
    const statusNames = ["順調（移動1）", "混雑（移動2）", "渋滞（移動4）"];
    lines.push("道路状態: " + statusNames[entry ? entry.status : 0]);
    if (state.day === 0) {
      lines.push("1日目なので全ての道路が順調です。");
    } else {
      const sourceDays = state.day === 1 ? [0] : [state.day - 2, state.day - 1];
      let total = 0;
      sourceDays.forEach((sourceDay) => {
        state.data.teams.forEach((team) => {
          const found = team.days[sourceDay].stayCounts.find((c) => c.pos === position);
          if (found) total += found.count;
        });
      });
      const players = problem.players;
      const traffic = (total / players).toFixed(2);
      lines.push(
        "交通量 = " + sourceDays.map((d) => d + 1 + "日目").join(" と ") +
        " の全チーム滞在数 " + total + " ÷ players " + players + " = " + traffic
      );
      lines.push(
        "判定: " + traffic + " が " + problem.busyThreshold + " 未満なら順調、" +
        problem.jammedThreshold + " 以上なら渋滞"
      );
    }
  }

  const spot = problem.spots.find((s) => s.pos === position);
  if (spot) {
    lines.push("スポット: 系列 " + spot.brand + " / 最大在庫 " + spot.stocks);
  }
  return lines.join("<br>");
}

canvas.addEventListener("click", (event) => {
  if (!state.data) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  let best = null;
  let bestDistance = Infinity;
  state.centers.forEach((center, position) => {
    const distance = (center.x - x) ** 2 + (center.y - y) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = position;
    }
  });
  if (best === null || bestDistance > (state.size * 1.2) ** 2) return;
  state.selectedCell = best;
  el("inspector").innerHTML = describeCell(best);
  render();
});

// ---------- 再生 ----------

el("step").addEventListener("input", (event) => {
  state.step = Number(event.target.value);
  render();
});

el("reset").onclick = () => {
  state.step = 0;
  syncStepRange();
  render();
};

function startPlayback() {
  if (state.timer) clearInterval(state.timer);
  if (!state.playing) return;
  state.timer = setInterval(() => {
    if (state.step >= maxStep()) {
      state.step = 0;
    } else {
      state.step += 1;
    }
    el("step").value = String(state.step);
    render();
  }, Math.max(16, BASE_INTERVAL_MS / state.speed));
}

el("play").onclick = () => {
  state.playing = !state.playing;
  el("play").textContent = state.playing ? "停止" : "再生";
  startPlayback();
};

el("speed").addEventListener("change", (event) => {
  state.speed = Number(event.target.value) || 1;
  startPlayback();
});

// ---------- キーボードで動かす ----------
//
// 試合の振り返りは「1歩ずつ送って、止まっている車を探す」作業に
// なります。マウスでスライダーを触るより、矢印キーのほうが速い。
//
//   空白 : 再生 / 一時停止      ← → : 1歩ずつ
//   Shift+← → : 10歩ずつ        ↑ ↓ : 日を変える
//   Home : 先頭へ               End : その日の最後へ
//   1〜9 : その番号の車だけ表示を切り替える
//   f    : 止まっている歩を光らせる
document.addEventListener("keydown", (event) => {
  if (event.target && /^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName)) {
    return;
  }
  const total = maxStep();
  const jump = event.shiftKey ? 10 : 1;
  let used = true;
  switch (event.key) {
    case " ":
      el("play").click();
      break;
    case "ArrowRight":
      state.step = Math.min(total, state.step + jump);
      break;
    case "ArrowLeft":
      state.step = Math.max(0, state.step - jump);
      break;
    case "ArrowUp":
      if (state.day > 0) { state.day -= 1; el("day").value = String(state.day); syncStepRange(); }
      break;
    case "ArrowDown":
      if (state.day < state.data.problem.daySteps.length - 1) {
        state.day += 1; el("day").value = String(state.day); syncStepRange();
      }
      break;
    case "Home":
      state.step = 0;
      break;
    case "End":
      state.step = total;
      break;
    case "f":
    case "F":
      state.showStill = !state.showStill;
      break;
    default:
      if (/^[1-9]$/.test(event.key)) {
        const seat = Number(event.key) - 1;
        if (state.stillOnly === seat) state.stillOnly = null;
        else state.stillOnly = seat;
      } else {
        used = false;
      }
  }
  if (used) {
    event.preventDefault();
    el("step").value = String(state.step);
    render();
  }
});

window.addEventListener("resize", () => {
  if (!state.data) return;
  layout();
  render();
});

buildDatasetSelect();
