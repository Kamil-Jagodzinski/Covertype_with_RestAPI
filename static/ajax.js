$(document).ready(function() {
    // Get data from from
    $('#submit-btn').click(function(e) {
        e.preventDefault();
        var model = $('#model').val();
        var Elevation = $('#Elevation').val();
        var Aspect = $('#Aspect').val();
        var Slope = $('#Slope').val();
        var hydrology_distance = $('#Horizontal_Distance_To_Hydrology').val();
        var hydrology_vertical = $('#Vertical_Distance_To_Hydrology').val();
        var roadways_distance = $('#Horizontal_Distance_To_Roadways').val();
        var hillshade_9am = $('#Hillshade_9am').val();
        var hillshade_noon = $('#Hillshade_Noon').val();
        var hillshade_3pm = $('#Hillshade_3pm').val();
        var fire_points_distance = $('#Horizontal_Distance_To_Fire_Points').val();
        var wilderness_area = $('#Wilderness_Area').val();
        var soil_type = $('#Soil_type').val();

        // Create data object
        var data = {
            'model': model,
            'Elevation': Elevation,
            'Aspect': Aspect,
            'Slope': Slope,
            'Horizontal_Distance_To_Hydrology': hydrology_distance,
            'Vertical_Distance_To_Hydrology': hydrology_vertical,
            'Horizontal_Distance_To_Roadways': roadways_distance,
            'Hillshade_9am': hillshade_9am,
            'Hillshade_Noon': hillshade_noon,
            'Hillshade_3pm': hillshade_3pm,
            'Horizontal_Distance_To_Fire_Points': fire_points_distance,
            'Wilderness_Area': wilderness_area,
            'Soil_type': soil_type
        };

        // Send data to backend
        $.ajax({
            url: 'http://127.0.0.1:5000/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(data),
            async: true,
            success: function(response) {
                // Display response
                $('#result').text("This is " +  response.pred);
            },
            error: function(xhr, status, error) {
                console.log(xhr.responseText);
                console.log(status);
                console.log(error);
            }
        });
    });
});
