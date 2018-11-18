require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

module.exports = class LinearRegression {
    constructor(features, labels, options = {learningRate: 0.1, iterations: 1000}) {
        this.features = this._processFeature(features);
        this.labels   = tf.tensor(labels);
        this.options  = options;

        this.weights = tf.zeros([2, 1]);
    }

    train() {
        for (let i = 0; i < this.options.iterations; i++) {
            this._gradientDescent()
        }
    }

    _gradientDescent() {
        const mse    = this.features.matMul(this.weights).sub(this.labels);
        const slopes = this.features.transpose().matMul(mse);

        const diff   = slopes.div(this.features.shape[0])
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

    _processFeature(features) {

        features = this._standardize(tf.tensor(features));

        features = tf.ones([features.shape[0], 1]).concat(features, 1);

        return features;
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
