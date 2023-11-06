import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
from sedona.register import SedonaRegistrator
from sedona.core.enums import FileDataSplitter, IndexType
from sedona.core.enums.join_build_side import JoinBuildSide

# Initialize Sedona
sedona_registrator = SedonaRegistrator.use()
sedona_registrator.registerAll()

# Import Sedona's functions
from sedona.core.geom.envelope import Envelope
from sedona.core.SpatialRDD import PolygonRDD, PointRDD

# Initialize a Spark session
spark = SparkSession.builder \
    .appName("SedonaFunctionTests") \
    .getOrCreate()

@pytest.fixture(scope="module")
def sample_data():
    # You can load sample data for testing here.
    # For example, create a DataFrame with geometries for testing purposes.
    sample_data = spark.createDataFrame([
        (1, "LINESTRING(0 0, 1 1, 1 0)"),
        (2, "POINT(21 52)"),
        (3, "POINT(1 1)"),
    ], ["id", "geometry"])
    return sample_data

def test_geometry_type(sample_data):
    # Test the GeometryType function
    result = sample_data.withColumn("geometrytype", expr("GeometryType(geometry)")).collect()
    assert result[0]["geometrytype"] == "LINESTRING"
    assert result[1]["geometrytype"] == "POINT"
    assert result[2]["geometrytype"] == "POINT"

def test_3d_distance(sample_data):
    # Test the ST_3DDistance function
    result = spark.sql("""
        SELECT ST_3DDistance(ST_GeomFromWKT("POINT Z (0 0 -5)"), ST_GeomFromWKT("POINT Z (1 1 -6)"))
    """).collect()
    assert abs(result[0][0] - 1.7320508075688772) < 1e-9

def test_add_point(sample_data):
    # Test the ST_AddPoint function
    result = sample_data.withColumn("modified_geometry", expr("ST_AddPoint(ST_GeomFromWKT(geometry), ST_GeomFromWKT('POINT(2 2)'), 1)")).collect()
    assert result[0]["modified_geometry"] == "LINESTRING (0 0, 2 2, 1 1, 1 0)"
    assert result[1]["modified_geometry"] == "POINT (21 52)"
    assert result[2]["modified_geometry"] == "POINT (1 1)"

def test_affine_transformation(sample_data):
    # Test the ST_Affine function
    result = spark.sql("""
        SELECT ST_Affine(ST_GeomFromWKT("POINT(1 1)"), 2, 1, 1, 2, 1, 2)
    """).collect()
    assert result[0][0] == "POINT (3 4)"

def test_angle_calculation(sample_data):
    # Test the ST_Angle function
    result = spark.sql("""
        SELECT ST_Angle(ST_GeomFromWKT('POINT(0 0)'), ST_GeomFromWKT('POINT(1 1)'), ST_GeomFromWKT('POINT(1 0)'), ST_GeomFromWKT('POINT(6 2)'))
    """).collect()
    assert abs(result[0][0] - 0.4048917862850834) < 1e-9

def test_area(sample_data):
    # Test the ST_Area function
    result = sample_data.withColumn("area", expr("ST_Area(geometry)")).collect()
    assert result[0]["area"] == 10
    assert result[1]["area"] == 201824850811.76245
    assert result[2]["area"] == 0

def test_area_spheroid(sample_data):
    # Test the ST_AreaSpheroid function
    result = spark.sql("""
        SELECT ST_AreaSpheroid(ST_GeomFromWKT('Polygon ((34 35, 28 30, 25 34, 34 35))')
    """).collect()
    assert abs(result[0][0] - 201824850811.76245) < 1e-9

def test_as_binary(sample_data):
    # Test the ST_AsBinary function
    result = spark.sql("""
        SELECT ST_AsBinary(ST_GeomFromWKT('POINT (1 1)'))
    """).collect()
    assert result[0][0] == "0101000000000000000000f87f000000000000f87f"

def test_as_ewkb(sample_data):
    # Test the ST_AsEWKB function
    result = spark.sql("""
        SELECT ST_AsEWKB(ST_SetSrid(ST_GeomFromWKT('POINT (1 1)'), 3021))
    """).collect()
    assert result[0][0] == "0101000020cd0b0000000000000000f03f000000000000f03f"

def test_as_ewkt(sample_data):
    # Test the ST_AsEWKT function
    result = spark.sql("""
        SELECT ST_AsEWKT(ST_SetSrid(ST_GeomFromWKT('POLYGON((0 0,0 1,1 1,1 0,0 0))'), 4326))
    """).collect()
    assert result[0][0] == "SRID=4326;POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"

def test_as_geojson(sample_data):
    # Test the ST_AsGeoJSON function
    result = spark.sql("""
        SELECT ST_AsGeoJSON(ST_GeomFromWKT('POLYGON((1 1, 8 1, 8 8, 1 8, 1 1))')
    """).collect()
    assert result[0][0] == '''{
  "type":"Polygon",
  "coordinates":[
    [[1.0,1.0],
      [8.0,1.0],
      [8.0,8.0],
      [1.0,8.0],
      [1.0,1.0]]
  ]
}'''

def test_as_gml(sample_data):
    # Test the ST_AsGML function
    result = spark.sql("""
        SELECT ST_AsGML(ST_GeomFromWKT('POLYGON((1 1, 8 1, 8 8, 1 8, 1 1))')
    """).collect()
    assert result[0][0] == "1.0,1.0 8.0,1.0 8.0,8.0 1.0,8.0 1.0,1.0"

def test_as_kml(sample_data):
    # Test the ST_AsKML function
    result = spark.sql("""
        SELECT ST_AsKML(ST_GeomFromWKT('POLYGON((1 1, 8 1, 8 8, 1 8, 1 1))')
    """).collect()
    assert result[0][0] == "1.0,1.0 8.0,1.0 8.0,8.0 1.0,8.0 1.0,1.0"

def test_as_text(sample_data):
    # Test the ST_AsText function
    result = spark.sql("""
        SELECT ST_AsText(ST_MakePointM(1.0, 1.0, 1.0))
    """).collect()
    assert result[0][0] == "POINT M(1 1 1)"

def test_azimuth(sample_data):
    # Test the ST_Azimuth function
    result = spark.sql("""
        SELECT ST_Azimuth(ST_POINT(0.0, 25.0), ST_POINT(0.0, 0.0))
    """).collect()
    assert abs(result[0][0] - 3.141592653589793) < 1e-9

def test_boundary(sample_data):
    # Test the ST_Boundary function
    result = spark.sql("""
        SELECT ST_AsText(ST_Boundary(ST_GeomFromWKT('POLYGON((1 1,0 0, -1 1, 1 1))'))
    """).collect()
    assert result[0][0] == "LINESTRING (1 1, 0 0, -1 1, 1 1)"

def test_bounding_diagonal(sample_data):
    # Test the ST_BoundingDiagonal function
    result = spark.sql("""
        SELECT ST_AsText(ST_BoundingDiagonal(ST_GeomFromWKT('POLYGON ((1 1 1, 3 3 3, 0 1 4, 4 4 0, 1 1 1))'))
    """).collect()
    assert result[0][0] == "LINESTRING Z(0 1 1, 4 4 4)"

def test_buffer(sample_data):
    # Test the ST_Buffer function
    result = spark.sql("""
        SELECT ST_AsText(ST_Buffer(ST_GeomFromWKT('POINT(0 0)'), 1))
    """).collect()
    assert result[0][0] == "POLYGON ((1 0, 0.9807852804032304 -0.195090

# Test case for ST_Degrees
def test_st_degrees():
    # Create a test DataFrame with radians values
    test_data = [(0.19739555984988044,)]
    test_df = spark.createDataFrame(test_data, ["radians"])

    # Calculate the expected result
    expected_result = test_df.select(expr("ST_Degrees(radians) as degrees")).first().degrees

    # Calculate the actual result
    actual_result = your_function_to_calculate_actual_result(test_df)

    # Assert that the actual result is equal to the expected result
    assert actual_result == pytest.approx(expected_result)

# Test case for ST_Difference
def test_st_difference():
    # Create a test DataFrame with two geometries
    test_data = [("POLYGON ((-3 -3, 3 -3, 3 3, -3 3, -3 -3))", "POLYGON ((0 -4, 4 -4, 4 4, 0 4, 0 -4))")]
    test_df = spark.createDataFrame(test_data, ["geometryA", "geometryB"])

    # Calculate the expected result
    expected_result = your_function_to_calculate_expected_result(test_df)

    # Calculate the actual result
    actual_result = your_function_to_calculate_actual_result(test_df)

    # Assert that the actual result is equal to the expected result
    assert actual_result == expected_result

# Test case for ST_Dimension
def test_st_dimension():
    # Create a test DataFrame with a geometry
    test_data = ["GEOMETRYCOLLECTION(LINESTRING(1 1,0 0),POINT(0 0))"]
    test_df = spark.createDataFrame(test_data, ["geometry"])

    # Calculate the expected result
    expected_result = test_df.select(expr("ST_Dimension(geometry) as dimension")).first().dimension

    # Calculate the actual result
    actual_result = your_function_to_calculate_actual_result(test_df)

    # Assert that the actual result is equal to the expected result
    assert actual_result == expected_result

# Test case for ST_Distance
def test_st_distance():
    # Create a test DataFrame with two geometries
    test_data = [("POINT(72 42)", "LINESTRING(-72 -42, 82 92)")]
    test_df = spark.createDataFrame(test_data, ["point", "lineString"])

    # Calculate the expected result
    expected_result = your_function_to_calculate_expected_result(test_df)

    # Calculate the actual result
    actual_result = your_function_to_calculate_actual_result(test_df)

    # Assert that the actual result is equal to the expected result
    assert actual_result == pytest.approx(expected_result)

# Test case for ST_DistanceSphere
def test_st_distance_sphere():
    # Create a test DataFrame with two points
    test_data = [("POINT (-0.56 51.3168)", "POINT (0.0 51.3168)")]
    test_df = spark.createDataFrame(test_data, ["point1", "point2"])

    # Calculate the expected result
    expected_result = your_function_to_calculate_expected_result(test_df)

    # Calculate the actual result
    actual_result = your_function_to_calculate_actual_result(test_df)

    # Assert that the actual result is equal to the expected result
    assert actual_result == pytest.approx(expected_result)

# Sample test cases for ST_MakeLine
def test_st_make_line_with_points():
    result = spark_sql_functions.st_make_line((1, 2), (3, 4))
    assert result == "LINESTRING(1 2,3 4)"

def test_st_make_line_with_linestrings():
    result = spark_sql_functions.st_make_line('LINESTRING(0 0, 1 1)', 'LINESTRING(2 2, 3 3)')
    assert result == "LINESTRING(0 0,1 1,2 2,3 3)"

# Sample test cases for ST_MakePolygon
def test_st_make_polygon():
    result = spark_sql_functions.st_make_polygon('LINESTRING(7 -1, 7 6, 9 6, 9 1, 7 -1)', ['LINESTRING(6 2, 8 2, 8 1, 6 1, 6 2)'])
    assert result == "POLYGON ((7 -1, 7 6, 9 6, 9 1, 7 -1), (6 2, 8 2, 8 1, 6 1, 6 2))"

# Test ST_Transform
def test_st_transform():
    input_polygon = PolygonRDD([((-71.01, 42.37), (-70.01, 44.37), (-69.01, 45.37))])
    transformed_polygon = input_polygon.ST_Transform("EPSG:4326", "EPSG:32649")
    assert transformed_polygon.count() == 1

# Test ST_Translate
def test_st_translate():
    input_point = PointRDD([(-71.01, 42.37), (-72.0, 43.0)])
    translated_point = input_point.ST_Translate(1, 2)
    assert translated_point.count() == 2

# Test ST_Union
def test_st_union():
    polygon1 = PolygonRDD([((-3, -3), (3, -3), (3, 3), (-3, 3), (-3, -3))])
    polygon2 = PolygonRDD([((1, -2), (5, 0), (1, 2), (1, -2))])
    union_result = polygon1.ST_Union(polygon2)
    assert union_result.count() == 1

# Test ST_VoronoiPolygons
def test_st_voronoi_polygons():
    input_points = PointRDD([(-3, -3), (3, 3), (0, 0)])
    voronoi_polygons = input_points.ST_VoronoiPolygons(0.0, None)
    assert voronoi_polygons.count() == 3

# Test ST_X, ST_Y
def test_st_x_y():
    point = PointRDD([(0.0, 25.0)])
    x = point.ST_X()
    y = point.ST_Y()
    assert x.collect()[0] == 0.0
    assert y.collect()[0] == 25.0

# Test ST_XMax, ST_XMin, ST_YMax, ST_YMin
def test_st_x_max_min_y_max_min():
    polygon = PolygonRDD([(-1, -11), (0, 10), (1, 11), (2, 12), (-1, -11)])
    x_max = polygon.ST_XMax()
    x_min = polygon.ST_XMin()
    y_max = polygon.ST_YMax()
    y_min = polygon.ST_YMin()
    assert x_max.collect()[0] == 2
    assert x_min.collect()[0] == -1
    assert y_max.collect()[0] == 12
    assert y_min.collect()[0] == -11

# Test ST_Z, ST_ZMax, ST_ZMin
def test_st_z_z_max_z_min():
    point = PointRDD([(0.0, 25.0, 11.0)])
    z = point.ST_Z()
    z_max = point.ST_ZMax()
    z_min = point.ST_ZMin()
    assert z.collect()[0] == 11.0
    assert z_max.collect()[0] == 11.0
    assert z_min.collect()[0] == 11.0

