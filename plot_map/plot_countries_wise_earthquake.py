from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


def plot_countries_wise_map(data):
    m = Basemap(projection='mill', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, lat_ts=20,
                resolution='c')

    longitudes = data["Longitude"].tolist()
    latitudes = data["Latitude"].tolist()
    x, y = m(longitudes, latitudes)

    fig = plt.figure(figsize=(12, 10))
    plt.title("All affected areas")
    m.plot(x, y, "o", markersize=2, color='blue')
    m.drawcoastlines()
    m.fillcontinents(color='coral', lake_color='aqua')
    m.drawmapboundary()
    m.drawcountries()
