/**
 * Method Registry
 *
 * Maps method names to factory functions for config-driven pipeline construction.
 * Researchers can register custom methods at runtime via registerMethod().
 *
 * Usage:
 *   // Built-in
 *   const pos = createMethod('POS', { sampleRate: 30 });
 *
 *   // Custom
 *   registerMethod('GREEN', (config) => ({
 *     name: 'GREEN',
 *     windowSize: Math.ceil(config.sampleRate * 1.6),
 *     needsTemporalNormalization: true,
 *     process: (rgb) => {
 *       const h = new Float32Array(rgb.g.length);
 *       const mu = mean(rgb.g);
 *       for (let i = 0; i < h.length; i++) h[i] = rgb.g[i] - mu;
 *       return h;
 *     }
 *   }));
 *   const green = createMethod('GREEN', { sampleRate: 30 });
 *
 *   TODO: note this and other functions here are reliant on the RGB window - some rPPG methods use the raw RGB or a single average.
 */

import type { WindowedPulseMethod } from './projection/types';
import { POS } from './projection/POS';
import { CHROM } from './projection/CHROM';
import { Green } from './projection/Green';
import { GRGB } from './projection/GRGB';
import { PBV } from './projection/PBV';

// ─── Types ──────────────────────────────────────────────────────────────────

// Configuration passed to method factories
export interface MethodConfig {
    sampleRate: number;
    windowMultiplier?: number;  // Default varies by method
}

// Factory function that creates a method instance from config
export type MethodFactory = (config: MethodConfig) => WindowedPulseMethod;

// ─── Registry ───────────────────────────────────────────────────────────────

const registry = new Map<string, MethodFactory>();

// Register built-in methods
const BUILTIN_METHODS: [string, new (sampleRate: number, windowMultiplier: number) => WindowedPulseMethod][] = [
    ['POS',   POS],
    ['CHROM', CHROM],
    ['Green', Green],
    ['GRGB',  GRGB],
    ['PBV',   PBV],
];

for (const [name, Ctor] of BUILTIN_METHODS) {
    registry.set(name, (config) => new Ctor(config.sampleRate, config.windowMultiplier ?? 1.6));
}
// ─── Public API ─────────────────────────────────────────────────────────────
// Create a method instance by name
export function createMethod(name: string, config: MethodConfig): WindowedPulseMethod {
    const factory = registry.get(name);
    if (!factory) {
        const available = Array.from(registry.keys()).join(', ');
        throw new Error(
            `Unknown pulse estimation method: '${name}'. ` +
            `Available: ${available}. ` +
            `Use registerMethod() to add custom methods.`
        );
    }
    return factory(config);
}

// Register custom pulse estimation method - overwrites any existing method with same name.
export function registerMethod(name: string, factory: MethodFactory): void {
    registry.set(name, factory);
}

// List all registered method names
export function getAvailableMethods(): string[] {
    return Array.from(registry.keys());
}
