import geopandas as gpd
import pandas as pd
import math
from pyproj import CRS

def build_taz_drive_connector(
    links_df, nodes_df,
    taz_polygon_df: gpd.GeoDataFrame, num_connectors_per_centroid: int = 3
):
    """
    build taz drive centroid connectors
    """

    # step 1
    # find nodes that have only two assignable/drive
    # geometries (not reference)

    # node_two_geometry_df = self.get_non_intersection_drive_nodes()
    node_two_geometry_df = _get_non_intersection_drive_nodes(links_df, nodes_df)

    # step 2
    # for each zone, find nodes that have only two assignable/drive
    # geometries (not reference) - good intersections

    taz_good_intersection_df = _get_nodes_in_zones(
        node_two_geometry_df, taz_polygon_df
    )

    # step 3
    # for zones with fewer than 4 good intersections
    # use drive nodes
    exclude_links_df = links_df[
        links_df.roadway.isin(
            ["motorway_link", "motorway", "trunk", "trunk_link"]
        )
    ].copy()

    drive_node_gdf = nodes_df[
        (nodes_df.drive_access == 1)
        & ~(
            nodes_df.osm_node_id.isin(
                exclude_links_df.u.tolist() + exclude_links_df.v.tolist()
            )
        )
    ].copy()

    taz_drive_node_df = _get_nodes_in_zones(drive_node_gdf, taz_polygon_df)

    # step 4
    # choose nodes for taz
    taz_centroid_gdf = taz_node_gdf.copy()
    taz_loading_node_df = get_taz_loading_nodes(
        taz_centroid_gdf,
        taz_good_intersection_df,
        taz_drive_node_df,
        num_connectors_per_centroid,
    )
    
    return taz_centroid_gdf, taz_loading_node_df
    



def _get_non_intersection_drive_nodes(links_df, nodes_df):
        """
        return nodes that have only two drivable geometries
        """
        drive_links_gdf = links_df[
            ~(
                links_df.roadway.isin(
                    ["motorway_link", "motorway", "trunk", "trunk_link", "service"]
                )
            )
            & (links_df.drive_access == 1)
        ].copy()

        a_geometry_count_df = (
            drive_links_gdf.groupby(["u", "shstGeometryId"])["shstReferenceId"]
            .count()
            .reset_index()
            .rename(columns={"u": "osm_node_id"})
        )

        b_geometry_count_df = (
            drive_links_gdf.groupby(["v", "shstGeometryId"])["shstReferenceId"]
            .count()
            .reset_index()
            .rename(columns={"v": "osm_node_id"})
        )

        node_geometry_count_df = pd.concat(
            [a_geometry_count_df, b_geometry_count_df], ignore_index=True, sort=False
        )

        node_geometry_count_df = (
            node_geometry_count_df.groupby(["osm_node_id", "shstGeometryId"])
            .count()
            .reset_index()
            .groupby(["osm_node_id"])["shstGeometryId"]
            .count()
            .reset_index()
        )

        node_two_geometry_df = node_geometry_count_df[
            node_geometry_count_df.shstGeometryId == 2
        ].copy()

        
        two_geometry_connected_to_node_slicer = nodes_df.osm_node_id.isin(node_two_geometry_df.osm_node_id.tolist())

        node_two_geometry_df = nodes_df[
            two_geometry_connected_to_node_slicer
        ].copy()

        nodde_not_two_geometru_df = nodes_df[
            ~two_geometry_connected_to_node_slicer
        ].copy()


        return node_two_geometry_df, nodde_not_two_geometru_df


def _get_nodes_in_zones(nodes_gdf, zones_gdf):
        """
        return nodes and the zones they are in

        Input:
            nodes_gdf: nodes geo data frame, points
            zones_gdf: zones geo data frame, polygons
        """
        polygon_buffer_gdf = zones_gdf.copy()

        polygon_buffer_gdf["geometry_buffer"] = polygon_buffer_gdf["geometry"].apply(
            lambda x: buffer1(x)
        )
        polygon_buffer_gdf.rename(
            columns={"geometry": "geometry_orig", "geometry_buffer": "geometry"},
            inplace=True,
        )
        nodes_in_zones_gdf = gpd.sjoin(
            nodes_gdf,
            polygon_buffer_gdf[["geometry", "taz_id"]],
            how="left",
            predicate="intersects",
        )

        return nodes_in_zones_gdf


def get_taz_loading_nodes(
    taz_centroid_df,
    good_intersection_df,
    drive_nodes_df,
    num_connectors_per_centroid,
):
    """
    for each zone, return chosen loading point
    """

    good_intersection_df["preferred"] = 1
    drive_nodes_df["preferred"] = 0

    all_load_nodes_df = pd.concat(
        [good_intersection_df, drive_nodes_df], sort=False, ignore_index=True
    )

    all_load_nodes_df = all_load_nodes_df[all_load_nodes_df["taz_id"].notnull()]

    all_load_nodes_df["ld_point"] = all_load_nodes_df["geometry"].apply(
        lambda x: list(x.coords)[0]
    )
    taz_centroid_df["c_point"] = taz_centroid_df["geometry"].apply(
        lambda x: list(x.coords)[0]
    )

    all_load_nodes_df = pd.merge(
        all_load_nodes_df.drop("geometry", axis=1),
        taz_centroid_df.drop("geometry", axis=1),
        how="left",
        on=["taz_id"],
    )

    all_load_nodes_df["distance"] = all_load_nodes_df.apply(
        lambda x: _haversine_distance(list(x.ld_point), list(x.c_point)), axis=1
    )

    # sort on preferred, distance
    all_load_nodes_df.sort_values(
        by=["preferred", "distance"], ascending=[False, True], inplace=True
    )

    all_load_nodes_df.drop_duplicates(
        subset=["osm_node_id", "taz_id"], inplace=True
    )

    all_load_nodes_df = _get_non_near_connectors(
        all_load_nodes_df, num_connectors_per_centroid, "taz_id"
    )

    return all_load_nodes_df


def _get_non_near_connectors(all_cc_link_gdf, num_connectors_per_centroid, zone_id: str):
    keep_cc_gdf = pd.DataFrame()

    for c in all_cc_link_gdf[zone_id].unique():
        # print("zone id {}".format(c))

        zone_cc_gdf = all_cc_link_gdf[all_cc_link_gdf[zone_id] == c].copy()

        centroid = zone_cc_gdf.c_point.iloc[0]

        # if the zone has less than 4 cc, keep all
        if len(zone_cc_gdf) <= num_connectors_per_centroid:
            keep_cc_gdf = keep_cc_gdf.append(zone_cc_gdf, sort=False, ignore_index=True)

        # if the zone has more than 4 cc
        else:
            zoneUnique = []

            zoneCandidate = zone_cc_gdf["ld_point"].to_list()
            # print("zone candidate {}".format(zoneCandidate))
            for point in zoneCandidate:
                # print("evaluate: {}".format(point))
                if len(zoneUnique) == 0:
                    zoneUnique += [point]
                else:
                    isDuplicate(point, centroid, zoneUnique)
                # print("zone unique {}".format(zoneUnique))
                if len(zoneUnique) == 4:
                    break

            zone_cc_gdf = zone_cc_gdf[
                zone_cc_gdf.ld_point.isin([tuple(z) for z in zoneUnique])
            ]

            keep_cc_gdf = keep_cc_gdf.append(zone_cc_gdf, sort=False, ignore_index=True)

    return keep_cc_gdf


def _haversine_distance(origin: list, destination: list, units="miles"):
    """
    Calculates haversine distance between two points

    Args:
    origin: lat/lon for point A
    destination: lat/lon for point B
    units: either "miles" or "meters"

    Returns: string
    """

    lon1, lat1 = origin
    lon2, lat2 = destination
    radius = 6378137  # meter

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(
        math.radians(lat1)
    ) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    d = {"meters": radius * c}  # meters
    d["miles"] = d["meters"] * 0.000621371  # miles

    return d[units]

def isDuplicate(a, b, zoneUnique):
    length = len(zoneUnique)
    # print("    unique zone unique length {}".format(length))
    for i in range(length):
        # print("           compare {} with zone unique {}".format(a, zoneUnique[i]))
        ang = getAngle(a, b, zoneUnique[i])

        if (ang < 45) | (ang > 315):
            return None

    zoneUnique += [a]

def getAngle(a, b, c):
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return ang + 360 if ang < 0 else ang


def buffer1(polygon):
    buffer_dist = 10
    poly_proj, crs_utm = project_geometry(polygon)
    poly_proj_buff = poly_proj.buffer(buffer_dist)
    poly_buff, _ = project_geometry(poly_proj_buff, crs=crs_utm, to_latlong=True)

    return poly_buff

def project_geometry(geometry, crs=None, to_crs=None, to_latlong=False):
    """
    Project a shapely geometry from its current CRS to another.
    If to_crs is None, project to the UTM CRS for the UTM zone in which the
    geometry's centroid lies. Otherwise project to the CRS defined by to_crs.
    Parameters
    ----------
    geometry : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        the geometry to project
    crs : dict or string or pyproj.CRS
        the starting CRS of the passed-in geometry. if None, it will be set to
        settings.default_crs
    to_crs : dict or string or pyproj.CRS
        if None, project to UTM zone in which geometry's centroid lies,
        otherwise project to this CRS
    to_latlong : bool
        if True, project to settings.default_crs and ignore to_crs
    Returns
    -------
    geometry_proj, crs : tuple
        the projected geometry and its new CRS
    """
    if crs is None:
        crs = CRS("epsg:4326")

    gdf = gpd.GeoDataFrame(geometry=[geometry], crs=crs)
    gdf_proj = project_gdf(gdf, to_crs=to_crs, to_latlong=to_latlong)
    geometry_proj = gdf_proj["geometry"].iloc[0]
    return geometry_proj, gdf_proj.crs


def project_gdf(gdf, to_crs=None, to_latlong=False):
    """
    Project a GeoDataFrame from its current CRS to another.
    If to_crs is None, project to the UTM CRS for the UTM zone in which the
    GeoDataFrame's centroid lies. Otherwise project to the CRS defined by
    to_crs. The simple UTM zone calculation in this function works well for
    most latitudes, but may not work for some extreme northern locations like
    Svalbard or far northern Norway.
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        the GeoDataFrame to be projected
    to_crs : dict or string or pyproj.CRS
        if None, project to UTM zone in which gdf's centroid lies, otherwise
        project to this CRS
    to_latlong : bool
        if True, project to settings.default_crs and ignore to_crs
    Returns
    -------
    gdf_proj : geopandas.GeoDataFrame
        the projected GeoDataFrame
    """
    if gdf.crs is None or len(gdf) < 1:
        raise ValueError("GeoDataFrame must have a valid CRS and cannot be empty")

    # if to_latlong is True, project the gdf to latlong
    if to_latlong:
        gdf_proj = gdf.to_crs(CRS("epsg:4326"))
        # utils.log(f"Projected GeoDataFrame to {settings.default_crs}")

    # else if to_crs was passed-in, project gdf to this CRS
    elif to_crs is not None:
        gdf_proj = gdf.to_crs(to_crs)
        # utils.log(f"Projected GeoDataFrame to {to_crs}")

    # otherwise, automatically project the gdf to UTM
    else:
        # if CRS.from_user_input(gdf.crs).is_projected:
        #   raise ValueError("Geometry must be unprojected to calculate UTM zone")

        # calculate longitude of centroid of union of all geometries in gdf
        avg_lng = gdf["geometry"].unary_union.centroid.x

        # calculate UTM zone from avg longitude to define CRS to project to
        utm_zone = int(math.floor((avg_lng + 180) / 6.0) + 1)
        utm_crs = (
            f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        )

        # project the GeoDataFrame to the UTM CRS
        gdf_proj = gdf.to_crs(utm_crs)
        # utils.log(f"Projected GeoDataFrame to {gdf_proj.crs}")

    return gdf_proj
