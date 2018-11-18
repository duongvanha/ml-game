const _ = require('lodash');

module.exports = class LinearRegression {
    constructor(features, labels, options = {learningRate: 0.1, iterations: 1000}) {
        this.features = features;
        this.labels   = labels;
        this.options  = options;

        this.b = 0;
        this.m = 0;
    }

    train() {
        for (let i = 0; i < this.options.iterations; i++) {
            this._gradientDescent()
        }
    }

    _gradientDescent() {
        const mse = this.features.map(item => this.m * item[0] + this.b);

        const mseB = (_.sum(mse.map((guess, i) => guess - this.labels[i][0])) * 2) / this.features.length;

        const mseM = (_.sum(mse.map((guess, i) =>
            -1 * this.features[i][0] * (this.labels[i][0] - guess),
        )) * 2 / this.features.length);

        this.m -= mseM * this.options.learningRate;
        this.b -= mseB * this.options.learningRate;
    }
};
