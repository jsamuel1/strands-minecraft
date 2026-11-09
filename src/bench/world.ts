/**
 * 🌍 One throwaway Minecraft server per episode — v3's World, same settings:
 * itzg image, fixed version and seed, world on tmpfs (nothing survives), easy,
 * survival, RCON for task setup. Port per episode so slow teardowns never collide.
 */
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { randomBytes } from 'node:crypto';

const run = promisify(execFile);

export const worldCfg = {
  image: process.env.BENCH_MC_IMAGE || 'itzg/minecraft-server:latest',
  version: process.env.BENCH_MC_VERSION || '1.21.11',
  seed: process.env.BENCH_SEED || '8675309',
};

/** The exact `docker run` argv. Pure — pinned by a test so the reset is auditable. */
export function dockerRunArgs(name: string, port: number, rconPort: number, cfg = worldCfg): string[] {
  return [
    'run', '-d', '--name', name,
    '-p', `127.0.0.1:${port}:25565`, '-p', `127.0.0.1:${rconPort}:25575`,
    '-e', 'EULA=TRUE', '-e', 'TYPE=VANILLA', '-e', `VERSION=${cfg.version}`, '-e', 'ONLINE_MODE=FALSE',
    '-e', 'MEMORY=2G', '-e', `SEED=${cfg.seed}`, '-e', 'DIFFICULTY=easy', '-e', 'MODE=survival',
    '-e', 'SPAWN_PROTECTION=0', '-e', 'VIEW_DISTANCE=6', '-e', 'ENABLE_RCON=true', '-e', 'RCON_PASSWORD=bench',
    '--tmpfs', '/data:size=3g', cfg.image,
  ];
}

export class World {
  readonly name = `bench-mc-${randomBytes(4).toString('hex')}`;
  constructor(readonly port: number, readonly rconPort: number) {}

  async start(timeoutS = 240): Promise<this> {
    await run('docker', dockerRunArgs(this.name, this.port, this.rconPort));
    const deadline = Date.now() + timeoutS * 1000;
    while (Date.now() < deadline) {
      const { stdout } = await run('docker', ['inspect', '-f', '{{.State.Health.Status}}', this.name]).catch(() => ({ stdout: '' }));
      if (stdout.trim() === 'healthy') return this;
      await new Promise((r) => setTimeout(r, 3000));
    }
    throw new Error(`${this.name} never became healthy`);
  }

  async rcon(command: string): Promise<string> {
    const { stdout } = await run('docker', ['exec', this.name, 'rcon-cli', command]);
    return stdout.trim();
  }

  async stop(): Promise<void> {
    await run('docker', ['rm', '-f', this.name]).catch(() => undefined);
  }
}
