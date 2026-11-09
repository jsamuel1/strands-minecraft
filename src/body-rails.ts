/**
 * 🩸 The body's own rails — pain, hunger, death — wired per connection.
 *
 * These lived inline in index.ts. They moved here when the brain got a second
 * home (the Python brain, which drives this body through the MCP sidecar in
 * sidecar.ts): both brains need the SAME pain/hunger/death facts, and two copies
 * of a death handler is how #35's repeat-grave rail went silent once already.
 *
 * Nothing here buys a model turn by itself. Every rail hands its facts to the
 * caller through `sinks` — `note` (rides in front of the next turn), `reflex`
 * (an urgent turn, gated by the caller's cooldown), `voice` (a briefing) — so
 * each brain keeps its own policy for what a turn costs.
 */
import type { LiveBody } from './body.js';
import { writePlace, recordDeath, deathContext, noteDamageSource } from './tools/memory.js';
import { readArmed } from './tools/helpers.js';

export interface BodyRailSinks {
  /** Queue a note for the mind (never pushed into a live history). */
  note: (text: string) => void;
  /** Ask for an urgent turn; the caller owns the cooldown gate. */
  reflex: (name: string, prompt: string) => void;
  /** Record what stretched a turn (issue #33's "why slow"). */
  turnEvent?: (what: string) => void;
  /** A briefing a human should HEAR (voice rail). */
  voice?: (source: string, text: string, importance: number) => void;
  /** What the mind was doing when it died — the running journey's goal, if any. */
  doing?: () => string | undefined;
  /** A fresh life: the caller resets its reflex cooldown. */
  onDeath?: () => void;
  /** Console visibility. Defaults to console.log. */
  log?: (text: string) => void;
  /**
   * Waypoint name for this body's last death. The primary keeps 'last_death';
   * a worker body passes its own ('digger_death') so two dying bodies never
   * overwrite each other's corpse-run (the fleet's rule).
   */
  deathWaypoint?: string;
}

/** Wire pain, hunger and death onto every connection the body makes. */
export function wireBodyRails(body: LiveBody, sinks: BodyRailSinks): void {
  const log = sinks.log ?? ((t: string) => console.log(t));
  const HUNGER_NOTE_AT = Number(process.env.HUNGER_NOTE_AT ?? 6);
  // Hunger is armed/disarmed with hysteresis, and the flag lives OUTSIDE
  // onEachBot on purpose: a reconnect re-fires 'health' with the same low
  // food, and a bot that re-announces its hunger after every signed-chat
  // kick is a bot nobody wants.
  let hungerArmed = true;
  body.onEachBot((b) => {
    b.on('entityHurt', (entity, source) => {
      // `entity` can be undefined: mineflayer looks the id up in bot.entities
      // and emits whatever it finds. Guard before the identity test — the same
      // missing check killed a soak from the worker rail.
      if (!entity || !b.entity || entity.id !== b.entity.id) return;
      // Snapshot the scene NOW — by the time the model answers, the zombie
      // has moved and the culprit may have despawned.
      const me = b.entity.position;
      // 1.20+ damage_event names the attacker directly; may be undefined
      // (fall, fire, cactus) or missing entirely on older servers.
      const attacker = source
        ? (source.type === 'player' ? (source.username ?? 'a player') : (source.name ?? 'something'))
        : undefined;
      const threats = Object.values(b.entities)
        .filter((e) => e.id !== b.entity.id && e.position && me.distanceTo(e.position) < 16
          && (e.type === 'hostile' || (e.kind ?? '').toLowerCase().includes('hostile') || e.type === 'player'))
        .sort((a, z) => me.distanceTo(a.position) - me.distanceTo(z.position))
        .slice(0, 5)
        .map((e) => `${e.name === 'player' ? (e.username ?? 'player') : (e.name ?? '?')} ${me.distanceTo(e.position).toFixed(1)}m away`);
      // #35: mineflayer's 'death' event carries no cause, so the LAST thing that
      // hit us is the only evidence a death site has about what kills there.
      if (attacker) noteDamageSource(attacker);
      log(`⚡ hurt${attacker ? ` by ${attacker}` : ''} — health ${(b.health ?? 0).toFixed(0)}/20${threats.length ? `, nearby: ${threats.join(', ')}` : ''}`);
      sinks.reflex(
        'hurt',
        `(reflex) You just TOOK DAMAGE${attacker ? ` from ${attacker.toUpperCase()}` : ''} — health ${(b.health ?? 0).toFixed(0)}/20, food ${b.food ?? '?'}/20. ` +
        (attacker
          ? `The attacker is identified: ${attacker}. `
          : threats.length
            ? `Likely culprits nearby: ${threats.join('; ')}. `
            : 'Nothing hostile in sight — could be fall, fire, lava, drowning or cactus. ') +
        'Handle the danger NOW, before anything else: fight back (attack_entity until=\'dead\' arms your strongest weapon itself and fights the whole duel), or retreat/pillar up if outmatched or low, deal with the environment (get out of lava/water, extinguish), and eat if safe. ' +
        'If a player is attacking you, one chat line to them is allowed.'
      );
    });
    // Starvation is SILENT: food drains with no dedicated event, sprint dies
    // below 7 drumsticks, and the first thing entityHurt hears about it is
    // starvation DAMAGE at food 0 — the pain reflex firing at the moment
    // eating can no longer prevent it. Catch the slide at the threshold
    // (HUNGER_NOTE_AT, default 6) instead. A note, not a reflex fork: hunger is
    // urgent-soon, not urgent-now. Once per hunger episode: re-arms only after
    // food recovers to 14+, so a meal of one berry doesn't buy a second warning.
    b.on('health', () => {
      if (process.env.REFLEX_DISABLED === 'true') return;
      const food = b.food;
      if (typeof food !== 'number') return; // pre-spawn tick
      if (hungerArmed && food <= HUNGER_NOTE_AT) {
        hungerArmed = false;
        log(`🍗 hungry — food ${food}/20`);
        sinks.note(
          `(system) You are HUNGRY — food ${food}/20. Below 7 you cannot sprint; at 0 you take starvation damage. ` +
          `Eat NOW — call eat with no item and it picks the best fit itself. Nothing safe in the bag? It will say what risky food you carry; otherwise get food the fastest way in reach: ` +
          `kill an animal and eat the drop, harvest a crop, or fish — and consider stocking a few meals while you're at it.`
        );
      } else if (!hungerArmed && food >= 14) {
        hungerArmed = true; // properly fed again — the next slide is news
      }
    });
    b.on('death', () => {
      sinks.onDeath?.(); // a fresh life deserves a fresh reflex
      hungerArmed = true; // respawn resets food to 20; the next slide is a new episode
      // Grab the position NOW — after respawn the entity stands at the spawn
      // point and the death spot is gone. Persist it too: 'where did I die?'
      // must survive a process restart.
      const p = b.entity?.position;
      log(`💀 died${p ? ` at ${Math.floor(p.x)}, ${Math.floor(p.y)}, ${Math.floor(p.z)}` : ''}`);
      let where = 'position unknown — it happened before spawn finished';
      // #35: 'last_death' is a NAME and writePlace replaces a name, so the
      // count comes back out of the death store, not out of this handler.
      const site = p
        ? recordDeath(p, {
            dimension: b.game?.dimension,
            // #35's remedy half: the conditions, so a cluster can name what its
            // deaths SHARE (no armour, after dark, the same job).
            ...deathContext(b, { doing: sinks.doing?.() }),
          })
        : undefined;
      // A fact that only ever reaches the mind cannot be verified by anyone
      // else (soak38) — the repeat grave is printed, too.
      if (site?.repeat) log(`💀 [grave] ${site.fact}`);
      if (p) {
        const place = writePlace(sinks.deathWaypoint ?? 'last_death', p, `died here (${b.game?.dimension ?? 'unknown dimension'})`, b.username ?? 'bot');
        where = `at (${place.x}, ${place.y}, ${place.z}) in ${b.game?.dimension ?? '?'} — saved as waypoint '${place.name}'`;
      }
      // Same rule as onRevive: never push into a possibly-mid-turn history.
      sinks.turnEvent?.('died');
      // #46: after a respawn the hands are EMPTY by definition.
      const armed = readArmed(b);
      // #35: on a REPEAT grave the stock corpse-run advice contradicts the site
      // fact, so it is replaced by the arithmetic.
      const corpseAdvice = site?.repeat
        ? `Only THIS life's drops are still there — the ${site.repeat.count - 1} earlier pile(s) at this spot are gone — and they are lying inside whatever has killed you ${site.repeat.count} times, so a corpse-run here is a re-entry, not an errand.`
        : 'Run back (or hire a worker for the corpse-run) — and think about what killed you before walking into it again.';
      sinks.note(`(system) You DIED and respawned. ${armed.line} Your items are on the ground ${where} and despawn in ~5 minutes. ${corpseAdvice}${site?.fact ? ` ${site.fact}` : ''}`);
      // Spoken to a human, so it must not promise a decision the mind has not
      // made (soak24): the facts only.
      sinks.voice?.('sentinel', `I just DIED ${where}. My gear is in that pile and despawns in about five minutes, and I respawned ${armed.armed ? `holding a ${armed.held}` : 'empty-handed'}.${site?.repeat ? ` That is death number ${site.repeat.count} within ${site.repeat.radius} blocks of the same spot.` : ''}`, 2);
    });
  });
}
