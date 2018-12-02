

import csv
from datetime import datetime, timedelta


ALL_TRADES_FILE = '../all_trades.csv'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

# Aggregation by hour.
ALL_TRADES_AGG_H_FILE = 'data/all_trades_h.csv'
START_HOUR = datetime.strptime('2015-09-25 13:00:00.000000', TIME_FORMAT)
DELTA_HOUR = timedelta(hours=1)

# Aggregation by minute.
ALL_TRADES_AGG_MIN_FILE = 'data/all_trades_min.csv'
START_MIN = datetime.strptime('2015-09-25 12:35:00.000000', TIME_FORMAT)
DELTA_MIN = timedelta(minutes=1)

# Frequency -> file with aggregates.
ALL_FREQUENCIES = {
    'MIN': ALL_TRADES_AGG_MIN_FILE,
    'H': ALL_TRADES_AGG_H_FILE,
}


# TODO Support a flag to aggregate by hour and other periods
def aggregate(in_file=ALL_TRADES_FILE, out_file=ALL_TRADES_AGG_MIN_FILE):
    aggs = []

    print('Starting aggregation')
    with open(in_file, 'r') as f_in:
        reader = csv.reader(f_in)

        next_dt = START_MIN  # Will aggregate everything that's happened by this point.
        delta = DELTA_MIN
        prices = []
        for timestamp, price in reader:
            dt = datetime.strptime(timestamp[:-3], TIME_FORMAT)
            price = float(price)

            if dt >= next_dt:
                # Flush prices for each time point in between.
                last_price = prices[-1]  # TODO Also track: mean and total?
                while dt >= next_dt:
                    aggs.append((next_dt, last_price))
                    next_dt += delta

                    if len(aggs) % (60 * 24 * 10) == 0:
                        print('Days:', len(aggs) // (60 * 24))

                # Reset prices.
                prices = []

            prices.append(price)

    print('Writing aggregates')
    with open(out_file, 'w') as f_out:
        writer = csv.writer(f_out)
        writer.writerows(aggs)
    print('Done')


def load_aggregates(frequency):
    filename = ALL_FREQUENCIES[frequency]
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return [(timestamp, float(price)) for timestamp, price in reader]


def main():
    aggregate()


if __name__ == '__main__':
    main()
