require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

module.exports = class LinearRegression {
    constructor(features, labels, options) {
        this.features   = this._processFeature(features);
        this.labels     = tf.tensor(labels);
        this.mseHistory = [];


        this.options = Object.assign({learningRate: 0.1, iterations: 1000, batchSize: 3}, options);
        this.weights = tf.zeros([this.features.shape[1], 1]);
    }

    train() {
        const {batchSize} = this.options;
        const batQuantity = Math.floor(this.features.shape[0] / batchSize);
        for (let i = 0; i < this.options.iterations; i++) {
            for (let j = 0; j < batQuantity; j++) {
                const features = this.features.slice([j * batchSize], [batchSize, -1]);
                const labels   = this.labels.slice([j * batchSize], [batchSize, -1]);
                this._gradientDescent(features, labels);
            }
            this._recordMSE();
            this._optimizeLearningRate()
        }
    }

    _gradientDescent(features, labels) {
        const mse    = features.matMul(this.weights).sub(labels);
        const slopes = features.transpose().matMul(mse);

        const diff   = slopes.div(features.shape[0])
              // .mul(2)
        ;
        this.weights = this.weights.sub(diff.mul(this.options.learningRate));
    }

    _standardize(features) {
        if (!this.mean) {
            const {mean, variance} = tf.moments(features, 0);
            this.mean              = mean;
            this.variance          = variance;
        }

        return features.sub(this.mean).div(this.variance.pow(0.5));
    }

    _recordMSE() {
        const mse = this.features
            .matMul(this.weights)
            .sub(this.labels)
            .pow(2)
            .sum()
            .div(this.features.shape[0])
            .get();

        this.mseHistory.unshift(mse);
    }

    _optimizeLearningRate() {
        if (this.mseHistory.length < 2) {
            return
        }

        this.options.learningRate *= this.mseHistory[0] > this.mseHistory[1] ? 0.5 : 1.05;
    }

    _processFeature(features) {

        features = this._standardize(tf.tensor(features));

        features = tf.ones([features.shape[0], 1]).concat(features, 1);

        return features;
    }

    predict(values) {
        return this._processFeature(values).matMul(this.weights);
    }

    test(testFeatures, testLabels) {
        testLabels   = tf.tensor(testLabels);
        testFeatures = this._processFeature(testFeatures);

        const predictions = testFeatures.matMul(this.weights);

        const res = testLabels.sub(predictions).pow(2).sum().get();

        const tot = testLabels.sub(testLabels.mean()).pow(2).sum().get();

        return 1 - (res / tot)
    }
};
