const nextJest = require('next/jest');

const createJestConfig = nextJest({ dir: './' });

const customConfig = {
  setupFilesAfterEnv: ['<rootDir>/jest.setup.ts'],
  testEnvironment: 'jsdom',
  moduleNameMapper: {
    '\\.(css|less|sass|scss)$': 'identity-obj-proxy',
  },
};

module.exports = createJestConfig(customConfig);
