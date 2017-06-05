var colour = d3.scaleOrdinal(d3.schemeCategory20b);

function click(d) {
	chart_group.transition()
		.duration(550)
		.tween("scale", function() {
			var xd = d3.interpolate(ascale.domain(), [d.x0, d.x1]),
			    yd = d3.interpolate(rscale.domain(), [d.y0, 1]),
			    yr = d3.interpolate(rscale.range(), [d.y0 ? 20 : 0, radius]);
	    	return function(t) { 
	    		ascale.domain(xd(t)); 
	    		rscale.domain(yd(t)).range(yr(t)); 
	    	}
		})
		.selectAll("path")
			.attrTween("d", function(d) { 
				//needs to return an interpolator function
				return function(t) {
					return arc(d);
				} 
			});
}
	
var width = 500,
    height = 400,
    radius = (Math.min(width, height) / 3);

var chart_group = d3.select("#content-div").append("svg")
		.attr("width", width)
		.attr("height", height)
	.append("g")
		.attr("transform", "translate(" + width / 2 + "," + (height / 2) + ")");


//Scaling funcs
var ascale = d3.scaleLinear()
    .range([0, 2 * Math.PI]);

var rscale = d3.scaleLinear()
    .range([0, radius]);

//Make hierarchal rects from json
var rt = d3.hierarchy(data).sum(function(d){return d.value});

//How hierachal data (icicle rects) is transformed to arcs
var arc = d3.arc()
    .startAngle(function(d) { 
    	return Math.max(0, Math.min(2*Math.PI, ascale(d.x0))); 
    })
    .endAngle(function(d) { 
    	return Math.max(0, Math.min(2*Math.PI, ascale(d.x1))); 
    })
    .innerRadius(function(d) { return rscale(d.y0) })
    .outerRadius(function(d) { return rscale(d.y1) });

function redrawSunburst(svg_group, data, arc, click) {
	//Make hierarchal rects from json
	rt = d3.hierarchy(data).sum(function(d){return d.value});

	var points = d3.partition()(rt).descendants();

	svg_group.selectAll("path").remove();
	svg_group.selectAll("path")
		.data(points)
		.enter().append("path")
			.attr("d", arc)
			.attr("fill", function(d) {
				//console.log(d3.arc()(d));
				return colour(d.data.label);
			})
			.on("click", click);
}

//init
redrawSunburst(chart_group, data, arc, click);
