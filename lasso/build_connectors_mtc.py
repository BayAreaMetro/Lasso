#%%

#%%
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from shapely.geometry import LineString
from scipy.spatial import cKDTree
import numpy as np
    

def reverse_linestring(geom):
    return LineString(geom.coords[::-1])

def rotate_point(point, angle):
    angle = angle * np.pi/180
    return point[0] * np.cos(angle) - point[1] * np.sin(angle), point[0] * np.sin(angle) + point[1] * np.cos(angle)

def dot_point(point1, point2):
    return (point1[0] * point2[0]) + (point1[1] * point2[1])

#%%
def split_points_into_thirds(df, initial_direction):
    #TODO generalise this so that it can split points into N evenly spaced bucketso
    points_dir_from_taz = df["X_net"] - df["X"], df["Y_net"] - df["Y"]

    rotate_150 = rotate_point(initial_direction, 150)
    rotate_90 = rotate_point(initial_direction, 90)
    in_group_1 = (dot_point(rotate_150, points_dir_from_taz) > 0) & (dot_point(rotate_90, points_dir_from_taz) > 0)
    rotate_neg_150 = rotate_point(initial_direction, -150)
    rotate_neg_90 = rotate_point(initial_direction, -90)
    in_group_2 = (dot_point(rotate_neg_150, points_dir_from_taz) > 0) & (dot_point(rotate_neg_90, points_dir_from_taz) > 0)
    
    df["group_third"] = 0
    df.loc[in_group_1, "group_third"] = 1
    df.loc[in_group_2, "group_third"] = 2

    return df

def get_geoms(closest_nodes):

    def get_linestring_from_row(row):
        return LineString([(row.X, row.Y), (row.X_net, row.Y_net)])
    closest_nodes["geometry"] = closest_nodes.apply(get_linestring_from_row, axis=1)
    return gpd.GeoDataFrame(closest_nodes, geometry='geometry', crs="EPSG:2875")

def get_two_way(links_df):
    links_df = links_df[["A", "B"]]
    
    ab = links_df["A"].astype(str) + "-" + links_df["B"].astype(str)
    ba = links_df["B"].astype(str) + "-" + links_df["A"].astype(str)

    return ab.isin(ba)


def connect_centroids(
    nodes_df: gpd.GeoDataFrame, links_df:gpd.GeoDataFrame, taz_centroid:gpd.GeoDataFrame, taz_zones: gpd.GeoDataFrame, parameters,
    maz_or_taz
):
    # taz_nodes = nodes_df[nodes_df["N"].isin(parameters.taz_N_list)]
    links_df["link_is_two_way"] = get_two_way(links_df) * 1
    print("proportion of links 2 way", links_df["link_is_two_way"].mean())
    links_df["link_is_two_way"] = links_df["link_is_two_way"] *-1

    #exclude nodes that are in intersections and already a centroid, 
    non_centroid_nodes = nodes_df[~nodes_df["N"].isin(taz_centroid["N"])]
    nodes_df_not_intersections, _ = get_non_intersection_drive_nodes(links_df, non_centroid_nodes)
    # we want to attatch to the lowest rank nodes, 
    nodes_df_not_intersections = join_ft_and_two_way_to_node(links_df, nodes_df_not_intersections, clip_upper=7)
    non_centroid_nodes = join_ft_and_two_way_to_node(links_df, non_centroid_nodes, clip_upper=7)

    #collect candidate nodes for building connector, change this to TAZ
    # get the x, y coordinates of the centroid of the shape for building the connector
    taz_area = taz_zones[[maz_or_taz, "geometry"]].merge(taz_centroid[["N", "X", "Y"]], left_on=maz_or_taz, right_on="N", how="inner")

    candidate_points = gpd.sjoin(taz_area, nodes_df_not_intersections[["N", "geometry", "X", "Y", "ft", "link_is_two_way", "county"]].rename(columns={"N":"N_Joined", "X": "X_net", "Y": "Y_net"}))
    
    # reverse ft so that when we sort ascending the ft=6 is first
    # after being sorted, it find the the closest ft6, if there are no ft6 then ft5 ect until
    # it finds one
    candidate_points["ft"] = -1 * candidate_points["ft"]
    links_to_be_added = []
    for taz_node_N, taz_node_candidate_links in candidate_points.groupby("N"):
        links = create_links(taz_node_N, taz_node_candidate_links)
        links["ft"] = abs(links["ft"])
        links_to_be_added.append(links)

    
    
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # if there were None, we will try again using all non centroid nodes
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    taz_with_no_links = taz_area[~taz_area["N"].isin(candidate_points["N"])]
    candidate_points = gpd.sjoin(taz_with_no_links, non_centroid_nodes[["N", "geometry", "X", "Y", "ft", "link_is_two_way", "county"]].rename(columns={"N":"N_Joined", "X": "X_net", "Y": "Y_net"}))

    candidate_points["ft"] = -1 * candidate_points["ft"]
    for taz_node_N, taz_node_candidate_links in candidate_points.groupby("N"):
        links = create_links(taz_node_N, taz_node_candidate_links)
        links["ft"] = abs(links["ft"])
        links_to_be_added.append(links)

    taz_with_no_links = taz_with_no_links[~taz_with_no_links["N"].isin(candidate_points["N"])]
    
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # if there is is still none we will match to the closest two-way node
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    taz_with_no_links["geometry"] = taz_with_no_links.apply(lambda row: Point(row["X"], row["Y"]), axis=1)
    # cant do sjoin nearest due to dependency issues, will change and check in future
    # connections_outside_taz_area = gpd.sjoin_nearest(centroids_with_no_connections, nodes_df_not_intersections, max_distance=parameters.max_length_centroid_connector_when_none_in_taz)
    non_centroid_nodes = non_centroid_nodes[(non_centroid_nodes["ft"] != 1) & (non_centroid_nodes["link_is_two_way"] != 0)]
    join_nodes_tree = cKDTree(non_centroid_nodes[["X", "Y"]].to_numpy())
    
    join_taz_nodes = []
    for index, row in enumerate(taz_with_no_links.itertuples(index=True)):
        join_distance, join_index = join_nodes_tree.query((row.X, row.Y))
        join_taz_nodes.append(non_centroid_nodes["N"].iloc[join_index])
    
    taz_with_no_links["join_taz_id"] = join_taz_nodes
    
    connections_outside_taz_area = pd.merge(
        taz_with_no_links, 
        non_centroid_nodes[["N", "geometry", "X", "Y", "ft", "county"]].rename(columns={"N":"N_Joined", "X": "X_net", "Y": "Y_net"}),
        left_on="join_taz_id", 
        right_on="N_Joined",
        how="left",
        validate="m:1"
    )
    connections_outside_taz_area["point_outside_taz"] = True

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # final post processing
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # reverse new links 
    new_links_gdf = pd.concat(links_to_be_added)
    new_links_gdf = pd.concat([new_links_gdf, connections_outside_taz_area])
    new_links_gdf = new_links_gdf.rename(columns={"N": "A", "N_Joined": "B"}).reset_index()
    new_links_gdf["CENTROID_NODE_ID_LINKED"] = new_links_gdf["A"]
    new_links_gdf = get_geoms(new_links_gdf)
    # right now we only have connector from TAZ to node, we need to go
    # backwards from node to TAZ, IE reverse the current directions

    new_links_reversed = new_links_gdf.copy()
    A = new_links_reversed["A"].copy()
    B = new_links_reversed["B"].copy()
    new_links_reversed["B"] = A
    new_links_reversed["A"] = B
    new_links_reversed["geometry"] = new_links_reversed["geometry"].apply(reverse_linestring)


    return pd.concat([new_links_gdf, new_links_reversed]).drop(columns = ["index_right", "taz", "X", "Y", "X_net", "Y_net", "squared_distance", "group_third"], errors="ignore")
    

    #exclude links 
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

def create_links(taz_node_N: int, taz_node_candidate_links: gpd.GeoDataFrame, sort_columns = ("link_is_two_way", "ft", "squared_distance")):
    

    taz_node_candidate_links.reset_index()

    X = taz_node_candidate_links.X.iloc[0]
    Y = taz_node_candidate_links.Y.iloc[0]
    taz_centroid_point = Point(X, Y) 
    
    # if we want distance we can take the suare root but there is no need
    # could do this using shapely but its all vectori= sed so it is fast
    taz_node_candidate_links["squared_distance"] = (taz_node_candidate_links["X_net"] - X)**2 + (taz_node_candidate_links["Y_net"] - Y)**2
    taz_node_candidate_links.sort_values(by=list(sort_columns), inplace=True, ascending=True)
    
    first_network_node_row = taz_node_candidate_links.iloc[0]

    X_net = first_network_node_row["X_net"]
    Y_net = first_network_node_row["Y_net"]
    distance = first_network_node_row["squared_distance"]**0.5

    direction_x, direction_y = (X_net-X) / distance, (Y_net-Y) / distance
    dir_point = direction_x, direction_y
    
    # to make sure we have 3 nodes in 3 different directions - must be more then 60 degrees from
    # first node, if we dont find any there are no nodes in that diretion
    taz_node_candidate_links = split_points_into_thirds(taz_node_candidate_links, dir_point)

    if sum(taz_node_candidate_links["group_third"]==1):
        second_network_node_row = taz_node_candidate_links[taz_node_candidate_links["group_third"]==1].iloc[0:1]
    else:
        second_network_node_row = None

    if sum(taz_node_candidate_links["group_third"]==2):
        third_network_node_row = taz_node_candidate_links[taz_node_candidate_links["group_third"]==2].iloc[0:1]
    else:
        third_network_node_row = None
    links = pd.concat([first_network_node_row.to_frame().T, second_network_node_row, third_network_node_row])
    
    links["taz_node_id"] = taz_node_N
    return links

def join_ft_and_two_way_to_node(links_df, nodes_df, clip_upper=6):
    
    all_ft_into_nodes = pd.concat(
        [
            links_df[["A", "ft", "link_is_two_way"]].rename(columns={"A":"N"}), 
            links_df[["B", "ft", "link_is_two_way"]].rename(columns={"A":"N"}),
        ]
    )
    highest_ft_attached_to_node = all_ft_into_nodes.groupby("N").agg({"ft":"max", "link_is_two_way":"max"})
    
    # highest_ft_attached_to_node["ft"] = highest_ft_attached_to_node[highest_ft_attached_to_node["ft"] <= 6]

    highest_ft_attached_to_node["ft"] = highest_ft_attached_to_node["ft"].clip(upper=clip_upper)
    
    return_nodes = pd.merge(nodes_df, highest_ft_attached_to_node, left_on="N", right_index=True) # should check this is inner merge 
    
    return_nodes = return_nodes[return_nodes["ft"] <= 5]

    return return_nodes

def get_non_intersection_drive_nodes(links_df, nodes_df):
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
    



# %%
