require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

module.exports = class LinearRegression {
    constructor(options) {
        this.options   = Object.assign({learningRate: 0.1, iterations: 1000, batchSize: 3}, options);
        this.optimizer = tf.train.sgd(this.options.learningRate);
        this.model     = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [3],
                    activation: 'relu',
                    units     : 4,
                }),
                tf.layers.dense({
                    units     : 1,
                    activation: 'relu',
                }),
            ],
        });
        this.model.compile({
            optimizer: this.optimizer,
            loss     : 'meanSquaredError',
        });

    }

    train(features, labels) {
        labels   = tf.tensor(labels);
        features = this._standardize(tf.tensor(features));

        return this.model.fit(features, labels, {epochs: this.options.iterations, shuffle: true})
    }

    _standardize(features) {
        if (!this.mean) {
            const {mean, variance} = tf.moments(features, 0);
            this.mean              = mean;
            this.variance          = variance;
        }

        return features.sub(this.mean).div(this.variance.pow(0.5));
    }

    predict(features) {
        features = this._standardize(tf.tensor(features));
        return this.model.predict(features)
    }

};

