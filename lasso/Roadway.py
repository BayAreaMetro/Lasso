import glob
import os

import geopandas as gpd
import pandas as pd

from geopandas import GeoDataFrame
from pandas import DataFrame

from network_wrangler import RoadwayNetwork
from .Parameters import Parameters
from .Logger import WranglerLogger


class ModelRoadwayNetwork(RoadwayNetwork):
    def __init__(
        self, nodes: GeoDataFrame, links: DataFrame, shapes: GeoDataFrame, parameters={}
    ):
        super().__init__(nodes, links, shapes)

        # will have to change if want to alter them
        self.parameters = Parameters(**parameters)

        self.links_metcouncil_df = None
        self.nodes_metcouncil_df = None
        self.shape_metcounciL_df = None
        ##todo also write to file
        # WranglerLogger.debug("Used PARAMS\n", '\n'.join(['{}: {}'.format(k,v) for k,v in self.parameters.__dict__.items()]))

    @staticmethod
    def read(
        link_file: str,
        node_file: str,
        shape_file: str,
        fast: bool = False,
        parameters={},
    ):
        # road_net =  super().read(link_file, node_file, shape_file, fast=fast)
        road_net = RoadwayNetwork.read(link_file, node_file, shape_file, fast=fast)

        m_road_net = ModelRoadwayNetwork(
            road_net.nodes_df,
            road_net.links_df,
            road_net.shapes_df,
            parameters=parameters,
        )

        return m_road_net

    @staticmethod
    def from_RoadwayNetwork(roadway_network_object, parameters={}):
        return ModelRoadwayNetwork(
            roadway_network_object.nodes_df,
            roadway_network_object.links_df,
            roadway_network_object.shapes_df,
            parameters=parameters,
        )

    def split_properties_by_time_period_and_category(self, properties_to_split=None):
        """
        Splits properties by time period, assuming a variable structure of

        Params
        ------
        properties_to_split: dict
             dictionary of output variable prefix mapped to the source variable and what to stratify it by
             e.g.
             {
                 'transit_priority' : {'v':'transit_priority', 'times_periods':DEFAULT_TIME_PERIOD_TO_TIME },
                 'traveltime_assert' : {'v':'traveltime_assert', 'times_periods':DEFAULT_TIME_PERIOD_TO_TIME },
                 'lanes' : {'v':'lanes', 'times_periods':DEFAULT_TIME_PERIOD_TO_TIME },
                 'price' : {'v':'price', 'times_periods':DEFAULT_TIME_PERIOD_TO_TIME ,'categories': DEFAULT_CATEGORIES},
                 'access' : {'v':'access', 'times_periods':DEFAULT_TIME_PERIOD_TO_TIME},
             }

        """
        import itertools

        if properties_to_split == None:
            properties_to_split = self.parameters.properties_to_split

        for out_var, params in properties_to_split.items():
            if params["v"] not in self.links_df.columns:
                raise ValueError(
                    "Specified variable to split: {} not in network variables: {}".format(
                        params["v"], str(self.links_df.columns)
                    )
                )
            if params.get("time_periods") and params.get("categories"):
                for time_suffix, category_suffix in itertools.product(
                    params["time_periods"], params["categories"]
                ):
                    self.links_df[
                        out_var + "_" + time_suffix + "_" + category_suffix
                    ] = self.get_property_by_time_period_and_group(
                        params["v"],
                        category=params["categories"][category_suffix],
                        time_period=params["time_periods"][time_suffix],
                    )
            elif params.get("time_periods"):
                for time_suffix in params["time_periods"]:
                    self.links_df[
                        out_var + "_" + time_suffix
                    ] = self.get_property_by_time_period_and_group(
                        params["v"],
                        category=None,
                        time_period=params["time_periods"][time_suffix],
                    )
            else:
                raise ValueError(
                    "Shoudn't have a category without a time period: {}".format(params)
                )

    def create_calculated_variables(self):
        """
        Params
        -------
        """

        self.calculate_area_type()
        self.calculate_county()
        self.calculate_centroid_connector()
        self.calculate_mpo()
        #self.calculate_assignment_group()
        #'self.calculate_roadway_class()
        self.add_counts()

    def calculate_county(self, network_variable="county"):
        """
        This uses the centroid of the geometry field to determine which county it should be labeled.
        This isn't perfect, but it much quicker than other methods.

        params
        -------

        """

        centroids_gdf = self.links_df.copy()
        centroids_gdf["geometry"] = centroids_gdf["geometry"].centroid

        county_gdf = gpd.read_file(self.parameters.county_shape)
        county_gdf = county_gdf.to_crs(epsg=RoadwayNetwork.EPSG)
        joined_gdf = gpd.sjoin(centroids_gdf, county_gdf, how="left", op="intersects")

        self.links_df[network_variable] = joined_gdf[
            self.parameters.county_variable_shp
        ]

    def calculate_area_type(
        self,
        network_variable="area_type",
        area_type_shape=None,
        area_type_shape_variable=None,
        area_type_codes_dict=None,
    ):
        """
        This uses the centroid of the geometry field to determine which area type it should be labeled.
        PER PRD
        ##TODO docstrings
        params
        -------

        """
        WranglerLogger.info("Calculating Area Type from Spatial Data")

        """
        Verify inputs
        """

        area_type_shape = (
            area_type_shape if area_type_shape else self.parameters.area_type_shape
        )

        if not area_type_shape:
            msg = "No area type shape specified"
            WranglerLogger.error(msg)
            raise ValueError(msg)
        if not os.path.exists(area_type_shape):
            msg = "File not found for area type shape: {}".format(area_type_shape)
            WranglerLogger.error(msg)
            raise ValueError(msg)

        area_type_shape_variable = (
            area_type_shape_variable
            if area_type_shape_variable
            else self.parameters.area_type_variable_shp
        )

        if not area_type_shape_variable:
            msg = "No area type shape varible specified"
            WranglerLogger.error(msg)
            raise ValueError(msg)

        area_type_codes_dict = (
            area_type_codes_dict
            if area_type_codes_dict
            else self.parameters.area_type_code_dict
        )
        if not area_type_codes_dict:
            msg = "No area type codes dictionary specified"
            WranglerLogger.error(msg)
            raise ValueError(msg)

        """
        Start actual process
        """
        centroids_gdf = self.links_df.copy()
        centroids_gdf["geometry"] = centroids_gdf["geometry"].centroid

        WranglerLogger.debug("Reading Area Type Shapefile {}".format(area_type_shape))
        area_type_gdf = gpd.read_file(area_type_shape)
        area_type_gdf = area_type_gdf.to_crs(epsg=RoadwayNetwork.EPSG)

        joined_gdf = gpd.sjoin(
            centroids_gdf, area_type_gdf, how="left", op="intersects"
        )

        joined_gdf[area_type_shape_variable] = (
            joined_gdf[area_type_shape_variable]
            .map(area_type_codes_dict)
            .fillna(10)
            .astype(int)
        )

        WranglerLogger.debug("Area Type Codes Used: {}".format(area_type_codes_dict))

        self.links_df[network_variable] = joined_gdf[area_type_shape_variable]

        WranglerLogger.info(
            "Finished Calculating Area Type from Spatial Data into variable: {}".format(
                network_variable
            )
        )

    def calculate_centroid_connector(
        self,
        network_variable="centroid_connector",
        as_integer=True,
        highest_taz_number=None,
    ):
        """
        Params
        ------
        network_variable: str
          variable that should be written to in the network
        as_integer: bool
          if true, will convert true/false to 1/0s
        """
        WranglerLogger.info("Calculating Centroid Connectors")
        """
        Verify inputs
        """
        highest_taz_number = (
            highest_taz_number
            if highest_taz_number
            else self.parameters.highest_taz_number
        )

        if not highest_taz_number:
            msg = "No highest_TAZ number specified in method variable or in parameters"
            WranglerLogger.error(msg)
            raise ValueError(msg)

        WranglerLogger.debug(
            "Calculating Centroid Connectors using highest TAZ number: {}".format(
                highest_taz_number
            )
        )

        if not network_variable:
            msg = "No network variable specified for centroid connector"
            WranglerLogger.error(msg)
            raise ValueError(msg)

        """
        Start actual process
        """
        self.links_df[network_variable] = False

        self.links_df.loc[
            (self.links_df["A"] <= highest_taz_number)
            | (self.links_df["B"] <= highest_taz_number),
            network_variable,
        ] = True

        if as_integer:
            self.links_df[network_variable] = self.links_df[network_variable].astype(
                int
            )
        WranglerLogger.info(
            "Finished calculating centroid connector variable: {}".format(
                network_variable
            )
        )

    def calculate_mpo(
        self,
        county_network_variable="county",
        network_variable="mpo",
        as_integer=True,
        mpo_counties=None,
    ):
        """
        Params
        ------
        county_variable: string
          name of the variable where the county names are stored.
        network_variable: string
          name of the variable that should be written to
        as_integer: bool
          if true, will convert true/false to 1/0s
        """
        WranglerLogger.info("Calculating MPO")
        """
        Verify inputs
        """
        county_network_variable = (
            county_network_variable
            if county_network_variable
            else self.parameters.county_network_variable
        )

        if not county_network_variable:
            msg = "No variable specified as containing 'county' in the network."
            WranglerLogger.error(msg)
            raise ValueError(msg)
        if county_network_variable not in self.links_df.columns:
            msg = "Specified county network variable: {} does not exist in network. Try running or debuging county calculation."
            WranglerLogger.error(msg)
            raise ValueError(msg)

        mpo_counties = mpo_counties if mpo_counties else self.parameters.mpo_counties

        if not mpo_counties:
            msg = "No MPO Counties specified in method call or in parameters."
            WranglerLogger.error(msg)
            raise ValueError(msg)

        WranglerLogger.debug("MPO Counties: {}".format(",".join(mpo_counties)))

        """
        Start actual process
        """

        mpo = self.links_df[county_network_variable].isin(mpo_counties)

        if as_integer:
            mpo = mpo.astype(int)

        self.links_df[network_variable] = mpo

        WranglerLogger.info(
            "Finished calculating MPO variable: {}".format(network_variable)
        )

    def calculate_assignment_group(
        self,
        network_variable="assignment_group",
        mrcc_roadway_class_shape=None,
        mrcc_shst_data=None,
        mrcc_roadway_class_variable_shp=None,
        mrcc_assgngrp_dict=None,
        widot_roadway_class_shape=None,
        widot_shst_data=None,
        widot_roadway_class_variable_shp=None,
        widot_assgngrp_dict=None,
        osm_assgngrp_dict=None,
    ):
        """
        Assignment Group is used in MetCouncil's traffic assignment to segment the volume/delay curves.
        Original source is from the MRCC data for the Minnesota: "route system" which is a roadway class
        For Wisconsin, it is from the Wisconsin DOT database, which has a variable called "roadway category"

        There is a crosswalk between the MRCC Route System and Wisconsin DOT --> Met Council Assignment group

        This method joins the  network with mrcc and widot roadway data by shst js matcher returns
        """

        WranglerLogger.info("Calculating Assignment Group")

        """
        Verify inputs
        """
        mrcc_roadway_class_shape = mrcc_roadway_class_shape if mrcc_roadway_class_shape else self.parameters.mrcc_roadway_class_shape
        if not mrcc_roadway_class_shape:
            msg = "'mrcc_roadway_class_shape' not found in method or lasso parameters.".format(mrcc_roadway_class_shape)
            WranglerLogger.error(msg)
            raise ValueError(msg)
        if not os.path.exists(mrcc_roadway_class_shape):
            msg = "'mrcc_roadway_class_shape' not found at following location: {}.".format(mrcc_roadway_class_shape)
            WranglerLogger.error(msg)
            raise ValueError(msg)

        widot_roadway_class_shape = widot_roadway_class_shape if widot_roadway_class_shape else self.parameters.widot_roadway_class_shape
        if not widot_roadway_class_shape:
            msg = "'widot_roadway_class_shape' not found in method or lasso parameters.".format(widot_roadway_class_shape)
            WranglerLogger.error(msg)
            raise ValueError(msg)
        if not os.path.exists(widot_roadway_class_shape):
            msg = "'widot_roadway_class_shape' not found at following location: {}.".format(widot_roadway_class_shape)
            WranglerLogger.error(msg)
            raise ValueError(msg)

        mrcc_shst_data = mrcc_shst_data if mrcc_shst_data else self.parameters.mrcc_shst_data
        if not mrcc_shst_data:
            msg = "'mrcc_shst_data' not found in method or lasso parameters.".format(mrcc_shst_data)
            WranglerLogger.error(msg)
            raise ValueError(msg)
        if not os.path.exists(mrcc_shst_data):
            msg = "'mrcc_shst_data' not found at following location: {}.".format(mrcc_shst_data)
            WranglerLogger.error(msg)
            raise ValueError(msg)

        widot_shst_data = widot_shst_data if widot_shst_data else self.parameters.widot_shst_data
        if not widot_shst_data:
            msg = "'widot_shst_data' not found in method or lasso parameters.".format(widot_shst_data)
            WranglerLogger.error(msg)
            raise ValueError(msg)
        if not os.path.exists(widot_shst_data):
            msg = "'widot_shst_data' not found at following location: {}.".format(widot_shst_data)
            WranglerLogger.error(msg)
            raise ValueError(msg)

        mrcc_roadway_class_variable_shp = mrcc_roadway_class_variable_shp if mrcc_roadway_class_variable_shp else self.parameters.mrcc_roadway_class_variable_shp
        if not mrcc_roadway_class_variable_shp:
            msg = "'mrcc_roadway_class_variable_shp' not found in method or lasso parameters."
            WranglerLogger.error(msg)
            raise ValueError(msg)

        widot_roadway_class_variable_shp = widot_roadway_class_variable_shp if widot_roadway_class_variable_shp else self.parameters.widot_roadway_class_variable_shp
        if not widot_roadway_class_variable_shp:
            msg = "'widot_roadway_class_variable_shp' not found in method or lasso parameters."
            WranglerLogger.error(msg)
            raise ValueError(msg)

        mrcc_assgngrp_dict = mrcc_assgngrp_dict if mrcc_assgngrp_dict else self.parameters.mrcc_assgngrp_dict
        if not mrcc_assgngrp_dict:
            msg = "'mrcc_assgngrp_dict' not found in method or lasso parameters."
            WranglerLogger.error(msg)
            raise ValueError(msg)

        widot_assgngrp_dict = widot_assgngrp_dict if widot_assgngrp_dict else self.parameters.widot_assgngrp_dict
        if not widot_assgngrp_dict:
            msg = "'widot_assgngrp_dict' not found in method or lasso parameters."
            WranglerLogger.error(msg)
            raise ValueError(msg)

        osm_assgngrp_dict = osm_assgngrp_dict if osm_assgngrp_dict else self.parameters.osm_assgngrp_dict
        if not osm_assgngrp_dict:
            msg = "'osm_assgngrp_dict' not found in method or lasso parameters.".format(osm_assgngrp_dict)
            WranglerLogger.error(msg)
            raise ValueError(msg)

        """
        Start actual process
        """

        WranglerLogger.debug("Calculating Centroid Connectors")
        self.calculate_centroid_connector()

        WranglerLogger.debug("Reading MRCC / Shared Streets Match CSV")
        # mrcc_shst_match_df = pd.read_csv()
        WranglerLogger.debug("Reading MRCC Shapefile: {}".format(mrcc_roadway_class_shape))
        mrcc_gdf = gpd.read_file(mrcc_roadway_class_shape)
        print("MRCC GDF Columns",mrcc_gdf.columns)
        #'LINK_ID', 'ROUTE_SYS', 'ST_CONCAT', 'geometry'
        mrcc_gdf["LINK_ID"] = range(1, 1 + len(mrcc_gdf))
        # returns shstreets dataframe with geometry ID, pp_link_id (which is the LINK_ID)

        #shstReferenceId,shstGeometryId,pp_link_id
        mrcc_shst_ref_df = pd.read_csv(mrcc_shst_data)
        print("mrcc shst ref df",mrcc_shst_ref_df.columns)

        widot_gdf = gpd.read_file(widot_roadway_class_shape)
        widot_gdf["LINK_ID"] = range(1, 1 + len(widot_gdf))
        print("WiDOT GDF Columns",widot_gdf.columns)
        widot_shst_ref_df = pd.read_csv(widot_shst_data)
        print("widot shst ref df",widot_shst_ref_df.columns)
        #join MRCC geodataframe with MRCC shared street return to get MRCC route_sys and shared street geometry id
        #
        # get route_sys from MRCC
        # end up with OSM data with MRCC attributes
        join_gdf = ModelRoadwayNetwork.get_attribute(
            self.links_df,
            "shstGeometryId",
            mrcc_shst_ref_df,
            mrcc_gdf,
            mrcc_roadway_class_variable_shp,
        )

        join_gdf = ModelRoadwayNetwork.get_attribute(
            join_gdf,
            "shstGeometryId",
            widot_shst_ref_df,
            widot_gdf,
            widot_roadway_class_variable_shp,
        )

        osm_asgngrp_crosswalk_df = pd.read_csv(osm_assgngrp_dict)
        mrcc_asgngrp_crosswalk_df = pd.read_csv(mrcc_assgngrp_dict)
        widot_asgngrp_crosswak_df = pd.read_csv(widot_assgngrp_dict)

        join_gdf = pd.merge(
            join_gdf,
            osm_asgngrp_crosswalk_df.rename(
                columns={"assignment_group": "assignment_group_osm"}
            ),
            how="left",
            on="roadway",
        )

        join_gdf = pd.merge(
            join_gdf,
            mrcc_asgngrp_crosswalk_df.rename(
                columns={"assignment_group": "assignment_group_mrcc"}
            ),
            how="left",
            on=mrcc_roadway_class_variable_shp,
        )

        join_gdf = pd.merge(
            join_gdf,
            widot_asgngrp_crosswak_df.rename(
                columns={"assignment_group": "assignment_group_widot"}
            ),
            how="left",
            on=widot_roadway_class_variable_shp,
        )

        def _set_asgngrp(x):
            try:
                if x.centroid_connector == 1:
                    return 9
                elif x.assignment_group_mrcc > 0:
                    return int(x.assignment_group_mrcc)
                elif x.assignment_group_widot > 0:
                    return int(x.assignment_group_widot)
                else:
                    return int(x.assignment_group_osm)
            except:
                return 0

        join_gdf[network_variable] = join_gdf.apply(lambda x: _set_asgngrp(x), axis=1)

        self.links_df[network_variable] = join_gdf[network_variable]

        WranglerLogger.info(
            "Finished calculating assignment group variable: {}".format(
                network_variable
            )
        )

    def calculate_roadway_class(
        self, network_variable="roadway_class", roadway_class_dict=None
    ):
        """
        roadway_class is a lookup based on assignment group

        """
        WranglerLogger.info("Calculating Roadway Class")

        """
        Verify inputs
        """
        roadway_class_dict = (
            roadway_class_dict
            if roadway_class_dict
            else self.parameters.roadway_class_dict
        )

        if not roadway_class_dict:
            msg = msg = "'roadway_class_dict' not found in method or lasso parameters."
            WranglerLogger.error(msg)
            raise ValueError(msg)

        """
        Start actual process
        """

        asgngrp_rc_num_crosswalk_df = pd.read_csv(roadway_class_dict)

        join_gdf = pd.merge(
            self.links_df,
            asgngrp_rc_num_crosswalk_df,
            how="left",
            on="assignment_group",
        )

        self.links_df[network_variable] = join_gdf[network_variable]

        WranglerLogger.info(
            "Finished calculating roadway class variable: {}".format(network_variable)
        )

    def add_variable_using_shst_reference(
        self,
        var_shst_csvdata=None,
        shst_csv_variable=None,
        network_variable=None,
        network_var_type=int,
        overwrite_existing=False,
    ):
        """
        join the network with data, via SHST API node match result
        """
        WranglerLogger.info(
            "Adding Variable {} using Shared Streets Reference from {}".format(
                network_variable, var_shst_csvdata
            )
        )

        var_shst_df = pd.read_csv(var_shst_csvdata)

        if "shstReferenceId" not in var_shst_df.columns:
            msg = "'shstReferenceId' required but not found in {}".format(var_shst_data)
            WranglerLogger.error(msg)
            raise ValueError(msg)

        join_gdf = pd.merge(
            self.links_df,
            var_shst_df[["shstReferenceId", shst_csv_variable]],
            how="left",
            on="shstReferenceId",
        )

        join_gdf[shst_csv_variable].fillna(0, inplace=True)

        if network_variable in self.links_df.columns and not overwrite_existing:
            join_gdf.loc[join_gdf[network_variable] > 0][
                network_variable
            ] = join_gdf.loc[self.links_df[network_variable] > 0][shst_csv_variable]
        else:
            join_gdf[network_variable] = join_gdf[shst_csv_variable].astype(
                network_var_type
            )

        self.links_df[network_variable] = join_gdf[network_variable]

        WranglerLogger.info(
            "Added variable: {} using Shared Streets Reference".format(network_variable)
        )

    def add_counts(
        self,
        mndot_count_shst_data=None,
        widot_count_shst_data=None,
        mndot_count_variable_shp=None,
        widot_count_variable_shp=None,
        network_variable="AADT",
    ):

        """
        join the network with count node data, via SHST API node match result
        """
        WranglerLogger.info("Adding Counts")

        """
        Verify inputs
        """

        mndot_count_shst_data = (
            mndot_count_shst_data
            if mndot_count_shst_data
            else self.parameters.mndot_count_shst_data
        )
        widot_count_shst_data = (
            widot_count_shst_data
            if widot_count_shst_data
            else self.parameters.widot_count_shst_data
        )
        mndot_count_variable_shp = (
            mndot_count_variable_shp
            if mndot_count_variable_shp
            else self.parameters.mndot_count_variable_shp
        )
        widot_count_variable_shp = (
            widot_count_variable_shp
            if widot_count_variable_shp
            else self.parameters.widot_count_variable_shp
        )

        for varname, var in {
            "mndot_count_shst_data": mndot_count_shst_data,
            "widot_count_shst_data": widot_count_shst_data,
        }.items():
            if not var:
                msg = "'{}' not found in method or lasso parameters.".format(varname)
                WranglerLogger.error(msg)
                raise ValueError(msg)
            if not os.path.exists(var):
                msg = "{}' not found at following location: {}.".format(varname, var)
                WranglerLogger.error(msg)
                raise ValueError(msg)

        for varname, var in {
            "mndot_count_variable_shp": mndot_count_variable_shp,
            "widot_count_variable_shp": widot_count_variable_shp,
        }.items():
            if not var:
                msg = "'{}' not found in method or lasso parameters.".format(varname)
                WranglerLogger.error(msg)
                raise ValueError(msg)

        """
        Start actual process
        """

        # Add Minnesota Counts
        self.add_variable_using_shst_reference(
            var_shst_csvdata=mndot_count_shst_data,
            shst_csv_variable=mndot_count_variable_shp,
            network_variable=network_variable,
            network_var_type=int,
            overwrite_existing=True,
        )

        # Add Wisconsin Counts, but don't overwrite Minnesota
        self.add_variable_using_shst_reference(
            var_shst_csvdata=widot_count_shst_data,
            shst_csv_variable=widot_count_variable_shp,
            network_variable=network_variable,
            network_var_type=int,
            overwrite_existing=False,
        )

        WranglerLogger.info(
            "Finished adding counts variable: {}".format(network_variable)
        )

    @staticmethod
    def read_match_result(path):
        """
        read the shst geojson match returns

        return shst dataframe

        reading lots of same type of file and concatenating them into a single DataFrame

        ##todo
        not sure why we need, but should be in utilities not this class
        """
        refId_gdf = DataFrame()
        refid_file = glob.glob(path)
        for i in refid_file:
            new = gpd.read_file(i)
            refId_gdf = pd.concat([refId_gdf, new], ignore_index=True, sort=False)
        return refId_gdf

    @staticmethod
    def get_attribute(
        links_df,
        join_key,  # either "shstReferenceId", or "shstGeometryId", tests showed the latter gave better coverage
        source_shst_ref_df,  # source shst refId
        source_gdf,  # source dataframe
        field_name,  # , # targetted attribute from source
    ):
        # join based on shared streets geometry ID
        # pp_link_id is shared streets match return
        # source_ink_id is mrcc
        print('source ShSt rename_variables_for_dbf columns', source_shst_ref_df.columns)
        print('source gdf columns', source_gdf.columns)
        # end up with OSM network with the MRCC Link ID
        # could also do with route_sys...would that be quicker?
        join_refId_df = pd.merge(
            links_df,
            source_shst_ref_df[[join_key, "pp_link_id", "score"]].rename(
                columns={"pp_link_id": "source_link_id", "score": "source_score"}
            ),
            how="left",
            on=join_key,
        )

        # joined with MRCC dataframe to get route_sys

        join_refId_df = pd.merge(
            join_refId_df,
            source_gdf[["LINK_ID", field_name]].rename(
                columns={"LINK_ID": "source_link_id"}
            ),
            how="left",
            on="source_link_id",
        )

        # more than one match, take the best score

        join_refId_df.sort_values(
            by=["model_link_id", "source_score"],
            ascending=True,
            na_position="first",
            inplace=True,
        )

        join_refId_df.drop_duplicates(
            subset=["model_link_id"], keep="last", inplace=True
        )

        # self.links_df[field_name] = join_refId_df[field_name]

        return join_refId_df[links_df.columns.tolist() + [field_name]]

    def roadway_standard_to_met_council_network(self, output_epsg=None):
        """
        rename and format roadway attributes to be consistent with what metcouncil's model is expecting.
        """

        WranglerLogger.info(
            "Renaming roadway attributes to be consistent with what metcouncil's model is expecting"
        )

        """
        Verify inputs
        """

        output_epsg = output_epsg if output_epsg else self.parameters.output_epsg

        """
        Start actual process
        """

        WranglerLogger.info("Calculating additional variables")
        self.create_calculated_variables()
        WranglerLogger.info("Splitting variables by time period and category")
        self.split_properties_by_time_period_and_category(
            {
                "transit_priority": {
                    "v": "transit_priority",
                    "time_periods": Parameters.DEFAULT_TIME_PERIOD_TO_TIME,
                    #'categories': Parameters.DEFAULT_CATEGORIES
                },
                "traveltime_assert": {
                    "v": "traveltime_assert",
                    "time_periods": Parameters.DEFAULT_TIME_PERIOD_TO_TIME,
                },
                "lanes": {
                    "v": "lanes",
                    "time_periods": Parameters.DEFAULT_TIME_PERIOD_TO_TIME,
                },
            }
        )

        self.links_metcouncil_df = self.links_df.copy()
        self.nodes_metcouncil_df = self.nodes_df.copy()
        self.shape_metcouncil_df = self.shapes_df.copy()

        WranglerLogger.info("Setting Coordinate Reference System")
        self.links_metcouncil_df.crs = RoadwayNetwork.CRS
        self.nodes_metcouncil_df.crs = RoadwayNetwork.CRS
        self.shape_metcouncil_df.crs = RoadwayNetwork.CRS
        self.links_metcouncil_df = self.links_metcouncil_df.to_crs(epsg=26915)
        self.nodes_metcouncil_df = self.nodes_metcouncil_df.to_crs(epsg=26915)
        self.shape_metcouncil_df = self.shape_metcouncil_df.to_crs(epsg=26915)

        self.nodes_metcouncil_df = self.nodes_metcouncil_df.reset_index()
        self.nodes_metcouncil_df.rename(columns={"index": "osm_node_id"}, inplace=True)

    def rename_variables_for_dbf(
        self, input_df, variable_crosswalk: str =None, output_variables: list =None
    ):
        """
        Rename attributes
        """
        WranglerLogger.info("Renaming variables so that they are DBF-safe")

        """
        Verify inputs
        """

        variable_crosswalk = (
            variable_crosswalk if variable_crosswalk else self.parameters.net_to_dbf
        )

        output_variables = (
            output_variables if output_variables else self.parameters.output_variables
        )

        """
        Start actual process
        """

        crosswalk_df = pd.read_csv(variable_crosswalk)
        WranglerLogger.debug(
            "Variable crosswalk: {} \n {}".format(
                self.parameters.net_to_dbf, crosswalk_df.info()
            )
        )
        net_to_dbf_dict = dict(zip(crosswalk_df["net"], crosswalk_df["dbf"]))

        dbf_name_list = []

        dbf_df = input_df.copy()

        # only write out variables that we specify
        # if variable is specified in the crosswalk, rename it to that variable
        for c in dbf_df.columns:
            if c in output_variables:
                try:
                    dbf_df.rename(columns={c: net_to_dbf_dict[c]}, inplace=True)
                    dbf_name_list += [net_to_dbf_dict[c]]
                except:
                    dbf_name_list += [c]

            if c == "geometry" and dbf_df["geometry"].geom_type[0] == "Point":
                dbf_df["X"] = dbf_df.geometry.apply(lambda g: g.x)
                dbf_df["Y"] = dbf_df.geometry.apply(lambda g: g.y)
                dbf_name_list += ["X", "Y"]

        WranglerLogger.debug("DBF Variables: {}".format(",".join(dbf_name_list)))

        return dbf_df[dbf_name_list]

    def write_roadway_as_shp(
        self,
        node_output_variables=None,
        link_output_variables=None,
        data_to_csv=True,
        data_to_dbf=False,
        output_link_shp=None,
        output_node_shp=None,
        output_link_csv=None,
        output_node_csv=None,
    ):
        """
        write out dbf/shp for cube
        write out csv in addition to shp with full length variable names
        """

        WranglerLogger.info("Writing Network as Shapefile")

        """
        Verify inputs
        """

        if not self.nodes_metcouncil_df:
            self.roadway_standard_to_met_council_network()

        link_output_variables = (
            link_output_variables
            if link_output_variables
            else [
                c
                for c in self.links_metcouncil_df.columns
                if c in self.parameters.output_variables
            ]
        )

        node_output_variables = (
            node_output_variables
            if node_output_variables
            else [
                c
                for c in self.nodes_metcouncil_df.columns
                if c in self.parameters.output_variables
            ]
        )

        #unless specified that all the data goes to the DBF, only output A and B
        dbf_link_output_variables = (
            dbf_link_output_variables if data_to_dbf else ["A", "B", "geometry"]
        )

        output_link_shp = (
            output_link_shp if output_link_shp else self.parameters.output_link_shp
        )

        output_node_shp = (
            output_node_shp if output_node_shp else self.parameters.output_node_shp
        )

        output_link_csv = (
            output_link_csv if output_link_csv else self.parameters.output_link_csv
        )

        output_node_csv = (
            output_node_csv if output_node_csv else self.parameters.output_node_csv
        )

        """
        Start Process
        """

        WranglerLogger.info("Renaming DBF Node Variables")
        nodes_dbf_df = self.rename_variables_for_dbf(
            self.nodes_metcouncil_df, output_variables=node_output_variables
        )
        WranglerLogger.info("Renaming DBF Link Variables")
        links_dbf_df = self.rename_variables_for_dbf(
            self.links_metcouncil_df, output_variables=dbf_link_output_variables
        )

        WranglerLogger.info("Writing Node Shapes:\n - {}".format(output_node_shp))
        nodes_dbf_df.to_file(output_node_shp)
        WranglerLogger.info("Writing Link Shapes:\n - {}".format(output_link_shp))
        links_dbf_df.to_file(output_link_shp)


        if data_to_csv:
            WranglerLogger.info(
                "Writing Network Data to CSVs:\n - {}\n - {}".format(
                    output_link_csv, output_node_csv
                )
            )
            self.links_metcouncil_df[link_output_variables].to_csv(
                output_link_csv, index=False
            )
            self.nodes_metcouncil_df[node_output_variables].to_csv(
                output_node_csv, index=False
            )
