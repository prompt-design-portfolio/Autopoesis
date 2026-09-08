/* Civitas UI (Part B §49).
 *
 * Thirteen screens over the versioned API, with no build step, no framework and no CDN. That is
 * not minimalism for its own sake: the page is served by the same process that serves the API, so
 * a deployment is one artifact and a Colab or air-gapped runtime renders the same UI as
 * production (§6, §33).
 *
 * Two rules run through every screen and are what the UI is actually for:
 *
 *   1. **Absence is not zero** (ARCHITECTURE §3.9). A metric the platform reports as `null`
 *      renders as "not available" with its reason. `fmt.metric` is the only path a number takes
 *      to the screen, and it cannot render `null` as `0`.
 *   2. **A failed gate withholds the result** (ARCHITECTURE §3.1). The benchmark screen renders
 *      the gates first and, when one failed, shows why the numbers are absent instead of
 *      rendering numbers the run did not earn.
 */
'use strict';

const API = '/api/v1';

/* ------------------------------------------------------------------ state */

const state = {
  workspaces: [],
  workspaceId: null,
  screen: 'overview',
  params: {},
  stream: null,
  lastSequence: 0,
  /* Which render is current. Two navigations in quick succession run their fetches
   * concurrently, and without this the *slower* one wins — leaving the screen you navigated away
   * from rendered over the one you asked for. Cheap to get wrong and invisible until someone
   * clicks twice. */
  renderToken: 0,
};

const KEY_STORE = 'civitas.apiKey';
const apiKey = {
  get() { try { return sessionStorage.getItem(KEY_STORE) || ''; } catch { return ''; } },
  set(v) { try { v ? sessionStorage.setItem(KEY_STORE, v) : sessionStorage.removeItem(KEY_STORE); }
           catch { /* private mode: the key simply does not persist */ } },
};

/* ------------------------------------------------------------------- http */

async function api(path, options = {}) {
  const headers = Object.assign({ 'accept': 'application/json' }, options.headers || {});
  const key = apiKey.get();
  if (key) headers['x-api-key'] = key;
  const response = await fetch(API + path, Object.assign({}, options, { headers }));
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch { /* not json */ }
    const error = new Error(`${response.status} ${detail}`);
    error.status = response.status;
    throw error;
  }
  return response.status === 204 ? null : response.json();
}

const ws = (path) => `/workspaces/${state.workspaceId}${path}`;

/* ------------------------------------------------------------- formatting */

const fmt = {
  /* The one path a measured number takes to the screen.
   *
   * `null` means the platform could not compute it, and the reason travels with it. Rendering
   * that as `0` would turn "not measurable" into a finding, which is the single most common way a
   * dashboard lies about an experiment. */
  metric(value, { digits = 4, reason = null, suffix = '' } = {}) {
    if (value === null || value === undefined) {
      const span = el('span', { class: 'na' }, reason ? `not available — ${reason}` : 'not available');
      return span;
    }
    const text = typeof value === 'number' ? value.toFixed(digits) : String(value);
    return document.createTextNode(text + suffix);
  },
  int(value) { return value === null || value === undefined ? null : Number(value).toLocaleString(); },
  usd(value) {
    if (value === null || value === undefined) return null;
    return '$' + Number(value).toFixed(Number(value) < 0.01 ? 6 : 4);
  },
  when(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : d.toISOString().replace('T', ' ').slice(0, 19);
  },
  id(value) { return value ? String(value).slice(0, 8) : '—'; },
};

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function cards(items) {
  return el('div', { class: 'cards' }, items.map(([k, v, opts]) => {
    const value = (v === null || v === undefined)
      ? el('div', { class: 'v na' }, opts && opts.reason ? `not available — ${opts.reason}` : 'not available')
      : el('div', { class: 'v' }, String(v));
    return el('div', { class: 'card' }, el('div', { class: 'k' }, k), value);
  }));
}

function table(columns, rows, render) {
  if (!rows.length) return el('p', { class: 'empty' }, 'Nothing here yet.');
  return el('div', { class: 'scroll' }, el('table', {},
    el('thead', {}, el('tr', {}, columns.map((c) => el('th', {}, c)))),
    el('tbody', {}, rows.map((row) => el('tr', {}, render(row).map((cell) =>
      cell instanceof HTMLTableCellElement ? cell : el('td', {}, cell)))))));
}

const wrapCell = (...kids) => el('td', { class: 'wrap' }, ...kids);
const pill = (text, kind) => el('span', { class: 'pill' + (kind ? ' ' + kind : '') }, text);
const link = (text, href) => el('a', { href }, text);
const artifactLink = (id, text) => link(text || fmt.id(id), `#/provenance?artifact=${id}`);
const episodeLink = (id) => link(fmt.id(id), `#/episode?id=${id}`);

/* ------------------------------------------------------------------ screens */

const SCREENS = [
  { group: 'Collective' },
  { id: 'overview',       title: 'Overview',        render: screenOverview },
  { id: 'projects',       title: 'Projects',        render: screenProjects },
  { id: 'tasks',          title: 'Tasks',           render: screenTasks },
  { id: 'episodes',       title: 'Episodes',        render: screenEpisodes },
  { group: 'Knowledge' },
  { id: 'artifacts',      title: 'Artifacts',       render: screenArtifacts },
  { id: 'retrieval',      title: 'Retrieval',       render: screenRetrieval },
  { id: 'provenance',     title: 'Provenance',      render: screenProvenance },
  { id: 'tools',          title: 'Tools',           render: screenTools },
  { group: 'Science' },
  { id: 'benchmarks',     title: 'Benchmarks',      render: screenBenchmarks },
  { id: 'metrics',        title: 'Metrics',         render: screenMetrics },
  { id: 'specialization', title: 'Specialization',  render: screenSpecialization },
  { id: 'institutions',   title: 'Institutions',    render: screenInstitutions },
  { group: 'Operations' },
  { id: 'cost',           title: 'Cost',            render: screenCost },
  { id: 'events',         title: 'Events',          render: screenEvents },
  /* Reached from a link rather than the nav. */
  { id: 'episode',        title: 'Episode',         render: screenEpisode, hidden: true },
  { id: 'project',        title: 'Project',         render: screenProject, hidden: true },
];

async function screenOverview(root) {
  const data = await api(ws('/overview'));
  state.lastSequence = Math.max(state.lastSequence, data.latest_sequence || 0);
  root.append(
    el('h1', {}, data.name),
    el('p', { class: 'lede' },
      'Everything below is a count of live rows. Archived artifacts are listed separately: '
      + '`memory_reset` archives rather than deletes, and one total would make an arm that lost '
      + 'its memory look like one that never had any.'),
    cards([
      ['projects', fmt.int(data.projects)],
      ['tasks', fmt.int(data.tasks)],
      ['ready / blocked', `${data.tasks_ready} / ${data.tasks_blocked}`],
      ['episodes', fmt.int(data.episodes)],
      ['succeeded', fmt.int(data.episodes_succeeded)],
      ['agents', fmt.int(data.agents)],
      ['artifacts', fmt.int(data.artifacts)],
      ['archived', fmt.int(data.artifacts_archived)],
      ['stale', fmt.int(data.artifacts_stale)],
      ['tools', fmt.int(data.tools)],
      ['events', fmt.int(data.events)],
      ['tokens', fmt.int(data.tokens_used)],
      ['cost', fmt.usd(data.cost_usd)],
      ['environment', data.environment_version],
    ]),
  );
}

async function screenProjects(root) {
  const projects = await api(ws('/projects'));
  root.append(
    el('h1', {}, 'Projects'),
    el('p', { class: 'lede' },
      'A high-level request becomes a project and a chain of tasks (§5). Every status below is '
      + 'reconstructed from the database on read, so it is identical after a restart (§32).'),
    table(['name', 'status', 'created', 'description'], projects, (p) => [
      link(p.name, `#/project?id=${p.id}`),
      pill(p.status),
      fmt.when(p.created_at),
      wrapCell(p.description || '—'),
    ]),
  );
}

async function screenProject(root) {
  const id = state.params.id;
  if (!id) { root.append(el('p', { class: 'empty' }, 'No project selected.')); return; }
  const [status, timeline] = await Promise.all([
    api(ws(`/projects/${id}/status`)),
    api(ws(`/projects/${id}/timeline`)),
  ]);
  root.append(
    el('h1', {}, status.name),
    el('p', { class: 'lede' }, 'Reconstructed from persisted rows alone (§32).'),
    cards([
      ['progress', status.progress === null ? null : (status.progress * 100).toFixed(0) + '%',
       { reason: 'the project has no tasks yet' }],
      ['tasks', `${status.tasks_completed} / ${status.tasks_total}`],
      ['blocked', fmt.int(status.tasks_blocked)],
      ['episodes', fmt.int(status.episodes)],
      ['succeeded', fmt.int(status.episodes_succeeded)],
      ['artifacts', fmt.int(status.artifacts)],
      ['validated', fmt.int(status.validated_artifacts)],
      ['open questions', fmt.int(status.open_questions)],
      ['tokens', fmt.int(status.tokens_used)],
      ['cost', fmt.usd(status.cost_usd)],
    ]),
    el('h2', {}, 'Stages'),
    table(['stage', 'status'], Object.entries(status.stages), ([family, s]) => [
      family, pill(s, s === 'completed' ? 'good' : s === 'blocked' ? 'warn' : null),
    ]),
    el('h2', {}, 'How the knowledge developed'),
    table(['at', 'type', 'title', 'validation', 'episode'], timeline, (row) => [
      fmt.when(row.at), row.type, wrapCell(artifactLink(row.artifact_id, row.title)),
      pill(row.validation_state),
      row.episode_id ? episodeLink(row.episode_id) : '—',
    ]),
  );
}

async function screenTasks(root) {
  const tasks = await api(ws('/tasks?limit=200'));
  root.append(
    el('h1', {}, 'Tasks'),
    el('p', { class: 'lede' },
      'The success criteria are not here and there is no field for them: an agent that could read '
      + 'its own evaluator would be self-certifying (§47).'),
    table(['title', 'family', 'status', 'attempts', 'difficulty', 'created'], tasks, (t) => [
      wrapCell(t.title), t.task_family || '—',
      pill(t.status, t.status === 'completed' ? 'good' : t.status === 'failed' ? 'bad' : null),
      fmt.int(t.attempts), t.difficulty?.toFixed?.(3) ?? '—', fmt.when(t.created_at),
    ]),
  );
}

async function screenEpisodes(root) {
  const episodes = await api(ws('/episodes?limit=200'));
  root.append(
    el('h1', {}, 'Episodes'),
    el('p', { class: 'lede' },
      'Bounded and temporary (§8). Nothing an episode thought is stored, so what you can inspect '
      + 'is what it did (§4).'),
    table(['id', 'arm', 'model', 'termination', 'tools', 'tokens', 'cost', 'probe', 'at'],
      episodes, (e) => [
        episodeLink(e.id),
        pill(e.experiment_arm),
        `${e.model_provider}/${e.model_name}`,
        pill(e.termination_reason || 'running',
          e.termination_reason === 'evaluator_success' ? 'good'
            : e.termination_reason === 'evaluator_failure' ? 'bad' : null),
        fmt.int(e.tool_calls_used), fmt.int(e.tokens_used), fmt.usd(e.cost_usd),
        e.is_benchmark_probe ? pill('probe', 'warn') : '',
        fmt.when(e.created_at),
      ]),
  );
}

async function screenEpisode(root) {
  const id = state.params.id;
  if (!id) { root.append(el('p', { class: 'empty' }, 'No episode selected.')); return; }
  const e = await api(ws(`/episodes/${id}`));
  root.append(
    el('h1', {}, 'Episode ' + fmt.id(e.id)),
    el('p', { class: 'lede' },
      'There is no reasoning trace here and none to add. §4 forbids persisting an episode’s '
      + 'private reasoning at all, so this shows configuration, budget, tool runs and output.'),
    cards([
      ['arm', e.experiment_arm],
      ['model', `${e.model_provider}/${e.model_name}`],
      ['model version', e.model_version || '—'],
      ['prompt version', e.system_prompt_version || '—'],
      ['retrieval policy', e.retrieval_policy_version || '—'],
      ['environment', e.environment_version],
      ['termination', e.termination_reason || 'running'],
      ['tool calls', fmt.int(e.tool_calls_used)],
      ['tokens', fmt.int(e.tokens_used)],
      ['cost', fmt.usd(e.cost_usd)],
      ['duration', e.duration_s === null ? null : e.duration_s.toFixed(2) + ' s',
       { reason: 'the episode has not finished' }],
      ['read / created', `${e.artifacts_read} / ${e.artifacts_created}`],
      ['duplicate failures', fmt.int(e.duplicate_failures)],
      ['benchmark probe', e.is_benchmark_probe ? 'yes' : 'no'],
      ['config hash', fmt.id(e.config_hash)],
    ]),
    el('h2', {}, 'Tool runs'),
    table(['at', 'arguments', 'ok', 'result', 'sandbox'], e.tool_runs, (r) => [
      fmt.when(r.started_at), el('code', {}, JSON.stringify(r.args)),
      r.succeeded ? pill('ok', 'good') : pill('rejected', 'bad'),
      wrapCell(r.stdout || '—'), r.sandbox_backend || '—',
    ]),
    el('h2', {}, 'Artifacts written'),
    table(['type', 'title', 'validation', 'archived'], e.artifacts, (a) => [
      a.type, wrapCell(artifactLink(a.id, a.title)), pill(a.validation_state),
      a.archived ? pill('archived', 'warn') : '',
    ]),
    el('h2', {}, 'Evaluations'),
    el('p', { class: 'lede' },
      'The evaluator writes these; no episode can (§47). The detail never contains ground truth.'),
    table(['scope', 'evaluator', 'succeeded', 'score', 'detail'], e.evaluations, (v) => [
      v.scope, `${v.evaluator_kind} ${v.evaluator_version}`,
      v.succeeded ? pill('yes', 'good') : pill('no', 'bad'),
      v.score?.toFixed?.(4) ?? '—',
      wrapCell(el('code', {}, JSON.stringify(v.detail))),
    ]),
  );
}

async function screenArtifacts(root) {
  const artifacts = await api(ws('/artifacts?limit=200'));
  root.append(
    el('h1', {}, 'Artifacts'),
    el('p', { class: 'lede' },
      'What survives the agents. A stale artifact is marked, not hidden: an agent that believed '
      + 'one is a measured outcome (§15, §48), not something to tidy away.'),
    table(['type', 'title', 'validation', 'env', 'stale', 'reads', 'utility', 'created'],
      artifacts, (a) => [
        a.type, wrapCell(artifactLink(a.id, a.title)),
        pill(a.validation_state,
          a.validation_state?.endsWith('confirmed') || a.validation_state === 'reproduced'
            ? 'good' : a.validation_state === 'refuted' ? 'bad' : null),
        a.environment_version || '—',
        a.is_stale ? pill('stale', 'warn') : '',
        fmt.int(a.times_read), a.downstream_utility?.toFixed?.(3) ?? '—',
        fmt.when(a.created_at),
      ]),
  );
}

async function screenRetrieval(root) {
  const form = el('form', { class: 'row' },
    el('input', { id: 'q', placeholder: 'query the collective', size: 46 }),
    el('select', { id: 'arm' },
      ['collective', 'solo', 'independent', 'shared_memory', 'collective_scrambled',
       'collective_no_negative', 'collective_no_provenance', 'memory_reset']
        .map((a) => el('option', { value: a }, a))),
    el('button', { type: 'submit' }, 'Retrieve'));
  const out = el('div');
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    out.replaceChildren(el('p', { class: 'empty' }, 'Retrieving…'));
    try {
      const result = await api(ws('/retrieval'), {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          query: form.querySelector('#q').value,
          experiment_arm: form.querySelector('#arm').value,
        }),
      });
      out.replaceChildren(
        el('p', { class: 'lede' },
          `policy ${result.policy_version} · ${result.results.length} shown`
          + (result.suppressed ? ` · ${result.suppressed} suppressed by the arm` : '')),
        table(['rank', 'type', 'title', 'score', 'why'], result.results, (r, i) => [
          String(r.rank ?? ''), r.type, wrapCell(artifactLink(r.artifact_id, r.title)),
          r.score?.toFixed?.(4) ?? '—', wrapCell(r.explanation || '—'),
        ]));
    } catch (error) {
      out.replaceChildren(el('p', { class: 'err' }, String(error.message)));
    }
  });
  root.append(
    el('h1', {}, 'Retrieval'),
    el('p', { class: 'lede' },
      'Run a query under any arm. The arm is enforced in the retriever, not in the prompt, so '
      + 'what a blinded arm cannot see is genuinely not returned (§21) — the count of suppressed '
      + 'results is shown so a blind arm is visibly blind rather than silently empty.'),
    form, out,
  );
}

async function screenProvenance(root) {
  const id = state.params.artifact;
  root.append(el('h1', {}, 'Provenance inspector'), el('p', { class: 'lede' },
    'Every claim traced to what it rests on (§13). Follow a title from any other screen to land '
    + 'here on that artifact.'));
  if (!id) {
    const artifacts = await api(ws('/artifacts?limit=100'));
    root.append(table(['type', 'title', 'validation'], artifacts, (a) => [
      a.type, wrapCell(artifactLink(a.id, a.title)), pill(a.validation_state),
    ]));
    return;
  }
  const data = await api(ws(`/artifacts/${id}/provenance`));
  const nodes = data.nodes || [];
  root.append(
    cards([
      ['depth reached', fmt.int(data.depth ?? nodes.length ? data.depth : null)],
      ['nodes', fmt.int(nodes.length)],
      ['edges', fmt.int((data.edges || []).length)],
      ['cycles', data.has_cycle ? 'yes' : 'no'],
    ]),
    el('h2', {}, 'Chain'),
    el('div', { class: 'prov' }, nodes.map((n) => el('div',
      { class: 'node depth-' + Math.min(4, n.depth ?? 0) },
      el('div', {}, pill(n.type), ' ', el('strong', {}, n.title || fmt.id(n.id))),
      el('div', { class: 'mono', style: 'color:var(--ink-dim)' },
        `${n.relation || 'root'} · ${n.validation_state || ''} · ${n.environment_version || ''}`),
      n.episode_id ? el('div', {}, 'by episode ', episodeLink(n.episode_id)) : null,
    ))),
  );
}

async function screenTools(root) {
  const tools = await api(ws('/tools'));
  root.append(
    el('h1', {}, 'Tool ecology'),
    el('p', { class: 'lede' },
      'Agents build tools for later agents (§16). A tool is discoverable only once a version has '
      + 'passed sandboxed validation — an untested tool is offered as untested, not withheld and '
      + 'not implied to work.'),
    table(['name', 'kind', 'builtin', 'versions', 'validation', 'runs', 'created'], tools, (t) => [
      wrapCell(t.name), t.kind, t.is_builtin ? pill('builtin') : '',
      fmt.int(t.version_count ?? 0),
      pill(t.validation_state || 'untested',
        t.validation_state === 'validated' ? 'good'
          : t.validation_state === 'failed' ? 'bad' : 'warn'),
      fmt.int(t.run_count ?? 0), fmt.when(t.created_at),
    ]),
  );
}

async function screenBenchmarks(root) {
  const experiments = await api(ws('/experiments'));
  root.append(
    el('h1', {}, 'Benchmarks'),
    el('p', { class: 'lede' },
      'Gates first. A failed gate does not annotate a result, it withholds it — there is nothing '
      + 'here for you to read as a number the run did not earn (ARCHITECTURE §3.1).'));
  if (!experiments.length) {
    root.append(el('p', { class: 'empty' }, 'No experiments recorded in this workspace.'));
    return;
  }
  for (const experiment of experiments) {
    const section = el('section', {}, el('h2', {}, experiment.name));
    root.append(section);
    let result;
    try {
      result = await api(`/experiments/${experiment.id}/result`);
    } catch (error) {
      section.append(el('p', { class: 'err' }, String(error.message)));
      continue;
    }
    section.append(table(['gate', 'passed', 'detail'], Object.entries(result.gates || {}),
      ([name, gate]) => [
        name, gate.passed ? pill('pass', 'good') : pill('fail', 'bad'),
        wrapCell(gate.detail || ''),
      ]));
    if (!result.gates_passed) {
      section.append(el('div', { class: 'withheld' },
        el('h3', {}, 'Metrics withheld'),
        el('p', {}, result.withheld_reason
          || `gates failed: ${(result.failed_gates || []).join(', ')}`),
        el('p', { class: 'note' },
          'This is the rule, not an error state: a result whose prerequisites failed is '
          + 'unreadable, not readable-with-caveats.')));
      continue;
    }
    section.append(el('pre', {}, JSON.stringify(result.metrics, null, 1)));
  }
}

async function screenMetrics(root) {
  const data = await api(ws('/metrics'));
  const entries = Object.entries(data.metrics || {});
  root.append(
    el('h1', {}, 'Metrics'),
    el('p', { class: 'lede' },
      `${(data.unavailable || []).length} of ${entries.length} metrics are not computable yet. `
      + 'They are shown as unavailable with a reason, never as zero — a dashboard that renders '
      + '"not measurable" as "measured zero" manufactures a finding (§48).'),
    table(['metric', 'value', 'unit', 'n', 'note'], entries, ([name, m]) => [
      name,
      el('td', {}, fmt.metric(m.value, { reason: m.unavailable_reason })),
      m.unit || '—', fmt.int(m.n) ?? '—', wrapCell(m.note || m.unavailable_reason || ''),
    ]),
  );
}

async function screenSpecialization(root) {
  const data = await api(ws('/specialization'));
  root.append(
    el('h1', {}, 'Specialization'),
    el('p', { class: 'lede' },
      'Division of cognitive labour: for each work type, how concentrated success is on one '
      + 'profile (§26). Reading this screen changes nothing — it measures without writing back.'),
    cards([['specialization index',
      data.specialization_index === null ? null : data.specialization_index.toFixed(4),
      { reason: data.index_unavailable_reason }]]),
    el('h2', {}, 'Profile performance by work type'),
    table(['profile', 'work type', 'n', 'success rate', '', 'tokens', 'tool calls', 'confident'],
      data.entries, (e) => [
        e.profile, e.work_type, fmt.int(e.n), e.success_rate.toFixed(4),
        el('td', {}, el('div', { class: 'bar' },
          el('span', { style: `width:${Math.round(e.success_rate * 100)}%` }))),
        fmt.int(Math.round(e.mean_tokens)), e.mean_tool_calls.toFixed(2),
        e.confident ? pill('yes', 'good') : pill('too few', 'warn'),
      ]),
  );
}

async function screenInstitutions(root) {
  const data = await api(ws('/institutions'));
  root.append(
    el('h1', {}, 'Institutions'),
    el('p', { class: 'lede' },
      'Self-modification passes a controlled experiment or it does not happen (§A2.2). A policy '
      + 'with no approving run is one the loader will not return, however good it looks.'),
    el('h2', {}, 'Gate'),
    table(['kind', 'proposed', 'approved', 'adoption rate', 'rolled back', 'mean improvement'],
      Object.entries(data.gate || {}), ([kind, g]) => [
        kind, fmt.int(g.proposed), fmt.int(g.approved),
        el('td', {}, fmt.metric(g.adoption_rate, { reason: 'nothing has been proposed' })),
        fmt.int(g.rolled_back),
        el('td', {}, fmt.metric(g.mean_attributed_improvement,
          { reason: 'no approval carried a matched comparison' })),
      ]),
    el('h2', {}, 'Policies'),
    table(['kind', 'name', 'version', 'status', 'approved', 'rolled back'], data.policies,
      (p) => [
        p.kind, wrapCell(p.name), String(p.version), pill(p.status),
        p.approved ? pill('approved', 'good') : pill('not approved', 'warn'),
        p.rolled_back ? pill('rolled back', 'bad') : '',
      ]),
    el('h2', {}, 'Reputation'),
    el('p', { class: 'lede' },
      'Multidimensional on purpose. A single score makes an agent that is unreliable-but-correct '
      + 'indistinguishable from one that is reliable-but-wrong (§29).'),
    el('div', {}, (data.reputations || []).map((r) => el('div', { class: 'card',
      style: 'margin-bottom:10px' },
      el('div', { class: 'k' }, r.subject_kind + ' ' + fmt.id(r.subject_id)),
      table(['dimension', 'value', 'n', 'why'], Object.entries(r.dimensions || {}),
        ([name, d]) => [
          name, el('td', {}, fmt.metric(d.value, { reason: d.unavailable_reason })),
          fmt.int(d.n) ?? '—', wrapCell(d.unavailable_reason || ''),
        ])))),
  );
}

async function screenCost(root) {
  const data = await api(ws('/cost'));
  const bucketTable = (rows) => table(['key', 'episodes', 'tokens', 'cost', 'mean cost'], rows,
    (b) => [
      b.key, fmt.int(b.episodes), fmt.int(b.tokens), fmt.usd(b.cost_usd),
      el('td', {}, fmt.metric(b.mean_cost_usd,
        { digits: 6, reason: 'no episode in this bucket recorded a cost' })),
    ]);
  root.append(
    el('h1', {}, 'Cost'),
    el('p', { class: 'lede' },
      `${data.unpriced_episodes} of ${data.episodes} episodes recorded no cost — offline and `
      + 'deterministic providers are genuinely free, so their mean cost is unknown rather than '
      + 'zero, and it is shown that way (§39).'),
    cards([
      ['total tokens', fmt.int(data.total_tokens)],
      ['total cost', fmt.usd(data.total_cost_usd)],
      ['episodes', fmt.int(data.episodes)],
      ['unpriced', fmt.int(data.unpriced_episodes)],
    ]),
    el('h2', {}, 'By day'), bucketTable(data.by_day),
    el('h2', {}, 'By model'), bucketTable(data.by_model),
    el('h2', {}, 'By arm'), bucketTable(data.by_arm),
  );
}

async function screenEvents(root) {
  const events = await api(ws('/events?limit=200'));
  root.append(
    el('h1', {}, 'Events'),
    el('p', { class: 'lede' },
      'The append-only log (§45). It is also the transport for this page’s live updates: '
      + 'each message carries its sequence, so a reconnecting browser resumes exactly where it '
      + 'stopped — a stream keyed on timestamps would skip or repeat rows sharing a millisecond.'),
    el('div', { id: 'event-rows' },
      table(['seq', 'type', 'actor', 'at', 'payload'], events, (e) => [
        String(e.sequence), pill(e.type), e.actor_kind, fmt.when(e.created_at),
        wrapCell(el('code', {}, JSON.stringify(e.payload))),
      ])),
  );
}

/* -------------------------------------------------------------- live stream */

function stopStream() {
  if (state.stream) { state.stream.close(); state.stream = null; }
  setLive(false);
}

function setLive(on) {
  const badge = document.getElementById('live');
  if (badge) { badge.className = 'live ' + (on ? 'on' : 'off'); }
}

function startStream() {
  stopStream();
  if (!state.workspaceId || typeof EventSource === 'undefined') return;
  /* EventSource cannot send headers, so the key travels as a query parameter on this one
   * same-origin request. It is the same secret either way; what changes is that it can appear in
   * an access log, which is why the server accepts it only on this route. */
  const key = apiKey.get();
  const url = `${API}/workspaces/${state.workspaceId}/stream?after=${state.lastSequence}`
    + (key ? `&api_key=${encodeURIComponent(key)}` : '');
  const source = new EventSource(url);
  state.stream = source;
  source.addEventListener('open', () => setLive(true));
  source.addEventListener('civitas', (message) => {
    let event;
    try { event = JSON.parse(message.data); } catch { return; }
    state.lastSequence = Math.max(state.lastSequence, event.sequence || 0);
    onLiveEvent(event);
  });
  source.addEventListener('error', () => setLive(false));
}

function onLiveEvent(event) {
  const rows = document.querySelector('#event-rows tbody');
  if (rows) {
    const row = el('tr', {},
      el('td', {}, String(event.sequence)), el('td', {}, pill(event.type)),
      el('td', {}, event.actor_kind), el('td', {}, fmt.when(event.at)),
      wrapCell(el('code', {}, JSON.stringify(event.payload))));
    rows.prepend(row);
    while (rows.children.length > 400) rows.lastChild.remove();
  }
  const badge = document.getElementById('live');
  if (badge) badge.textContent = 'live · ' + event.sequence;
}

/* ------------------------------------------------------------------ routing */

function parseHash() {
  const raw = (location.hash || '#/overview').slice(2);
  const [path, query] = raw.split('?');
  const params = {};
  new URLSearchParams(query || '').forEach((v, k) => { params[k] = v; });
  return { screen: path || 'overview', params };
}

function renderNav() {
  const nav = document.getElementById('nav');
  nav.replaceChildren();
  for (const item of SCREENS) {
    if (item.group) { nav.append(el('div', { class: 'grp' }, item.group)); continue; }
    if (item.hidden) continue;
    nav.append(el('a', {
      href: `#/${item.id}`,
      class: item.id === state.screen ? 'active' : '',
    }, item.title));
  }
}

async function route() {
  const { screen, params } = parseHash();
  const token = ++state.renderToken;
  const current = () => state.renderToken === token;

  state.screen = screen;
  state.params = params;
  renderNav();
  const main = document.getElementById('main');
  main.replaceChildren(el('p', { class: 'empty' }, 'Loading…'));
  const entry = SCREENS.find((s) => s.id === screen);
  if (!entry) {
    main.replaceChildren(el('h1', {}, 'Not a screen'),
      el('p', { class: 'empty' }, `No screen named "${screen}".`));
    return;
  }
  if (!state.workspaceId) {
    main.replaceChildren(el('h1', {}, entry.title),
      el('p', { class: 'empty' },
        'Select a workspace. If the list is empty, set an API key or create one with '
        + '`civitas bootstrap`.'));
    return;
  }
  const root = el('div');
  try {
    await entry.render(root);
    if (current()) main.replaceChildren(root);
  } catch (error) {
    if (!current()) return;
    main.replaceChildren(
      el('h1', {}, entry.title),
      el('p', { class: 'err' }, String(error.message)),
      error.status === 401 || error.status === 403
        ? el('p', {}, 'Set an API key with the button in the header.')
        : null);
  }
}

/* --------------------------------------------------------------- bootstrap */

async function loadWorkspaces() {
  const select = document.getElementById('workspace-select');
  select.replaceChildren();
  try {
    state.workspaces = await api('/workspaces');
  } catch (error) {
    state.workspaces = [];
    select.append(el('option', {}, error.status === 401 ? 'set an API key' : 'unavailable'));
    return;
  }
  if (!state.workspaces.length) {
    select.append(el('option', {}, 'no workspaces'));
    return;
  }
  for (const workspace of state.workspaces) {
    select.append(el('option', { value: workspace.id }, workspace.name));
  }
  const remembered = (() => { try { return localStorage.getItem('civitas.workspace'); }
                              catch { return null; } })();
  const chosen = state.workspaces.find((w) => w.id === remembered) || state.workspaces[0];
  select.value = chosen.id;
  setWorkspace(chosen);
}

function setWorkspace(workspace) {
  state.workspaceId = workspace.id;
  state.lastSequence = 0;
  try { localStorage.setItem('civitas.workspace', workspace.id); } catch { /* private mode */ }
  const env = document.getElementById('env');
  if (env) env.textContent = workspace.environment_version || '';
  startStream();
}

function wireHeader() {
  document.getElementById('workspace-select').addEventListener('change', (event) => {
    const workspace = state.workspaces.find((w) => w.id === event.target.value);
    if (workspace) { setWorkspace(workspace); route(); }
  });
  const dialog = document.getElementById('key-dialog');
  document.getElementById('key-button').addEventListener('click', () => {
    document.getElementById('key-input').value = apiKey.get();
    dialog.showModal();
  });
  dialog.addEventListener('close', async () => {
    if (dialog.returnValue === 'save') apiKey.set(document.getElementById('key-input').value.trim());
    else if (dialog.returnValue === 'clear') apiKey.set('');
    else return;
    await loadWorkspaces();
    route();
  });
  window.addEventListener('hashchange', route);
}

async function main() {
  wireHeader();
  await loadWorkspaces();
  await route();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', main);
else main();

/* Exposed for the browser test, which drives the real page rather than a mock of it. */
window.civitas = { state, fmt, api, route, SCREENS };
