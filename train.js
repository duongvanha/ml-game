const _ = require('lodash');

function getDistance(pointA, pointB) {
    return _.chain(pointA)
        .zip(pointB)
        .map(([a, b]) => (a - b) ** 2)
        .sum()
        .value() ** 0.5
}

module.exports = function (data, point, k) {
    return _.chain(data)
        .map(item => [getDistance(_.initial(item), point), _.last(item)])
        .sortBy(item => item[0])
        .slice(0, k)
        .countBy(row => row[1])
        .toPairs()
        .sortBy(row => row[1])
        .last()
        .first()
        .parseInt()
        .value()
}
