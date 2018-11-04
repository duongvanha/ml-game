const _ = require('lodash');

function getDistance(pointA, pointB) {
    return _.chain(pointA)
        .zip(pointB)
        .map((current) => (current[0] - current[1]) ** 2)
        .sum()
        .value() ** 0.5
}

function normalization(data, countColumn) {
    const dataClone = _.cloneDeep(data)
    for (let i = 0; i < countColumn; i++) {
        const column = dataClone.map(row => row[i])
        const max = _.max(column);
        const min = _.min(column);

        for (let j = 0; j < column.length; j++) {
            // if min == max infinity
            dataClone[j][i] = (dataClone[j][i] - min) / (max - min)

        }
    }

    return dataClone
}

module.exports = function (data, point, k) {
    return _.chain(normalization(data, 0))
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
