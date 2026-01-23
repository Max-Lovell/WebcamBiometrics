export default {
    preset: 'ts-jest',
    testEnvironment: 'node', // Use node environment for canvas/math utils
    transform: {
        // Replicate the original project's transform settings
        '^.+\\.tsx?$': ['ts-jest', {
            diagnostics: {
                ignoreCodes: ['TS151001']
            }
        }]
    },
    moduleNameMapper: {
        // Handle ESM imports if necessary
        '^(\\.{1,2}/.*)\\.js$': '$1',
    }
};
