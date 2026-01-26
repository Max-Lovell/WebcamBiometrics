export default {
    preset: 'ts-jest',
    testEnvironment: 'node',
    transform: {
        '^.+\\.tsx?$': ['ts-jest', {
            diagnostics: {
                ignoreCodes: ['TS151001']
            }
        }]
    },
    moduleNameMapper: {
        '^(\\.{1,2}/.*)\\.js$': '$1',
    }
};
