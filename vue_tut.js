function getIndex(arr, id) {
	return arr.findIndex(function(o){ return o.id === id });
}

function haveOneEmpty(rules_arr) {
	var count = 0;
	var ids = [];
	rules_arr.map(function(rule_obj, i) {
		if (rule_obj.text === "") {
			if (count >= 1) rules_arr.splice(i, 1);	
			else count += 1;
		}
		ids.push(rule_obj.id);
	});
	if (count === 0) rules_arr.push( {id:Math.max(...ids)+1, text: ""} );
	return rules_arr;
}

function insertCategoryIntoLabel(data, cat, label, s) {
	if (data.label === label) {
		insertCategoryHere(data, cat, s);
		return "done"
	}
	else if (data.children !== undefined) {
		var current_status = "notfound"
		for (var i=0; i<data.children.length; i++) {		
			var child = data.children[i];
			var child_status = insertCategoryIntoLabel(child, cat, label, s);
			if (child_status === "done") {
				current_status = "done";
				break;
			}
		}
		return current_status;
	} else {
		return "notfound";
	}
}

function insertCategoryHere(node, cat, s) {
	var new_node = {label: cat, children: []};
	var new_children = [];
	node.children.map(function(child, i) {
		if (child.label.includes(s)) {
			new_node.children.push(child);
		} else {
			new_children.push(child);
		}
	});
	node.children = new_children;
	node.children.push(new_node);
}

function insertCategoryIntoData(rt, cat, s, data) {
	var counts_map = {};

	rt.descendants().map(function(node) {
		if (node.parent !== null) {
			if (node.data.label.includes(s)) {
				var curr_cat = node.parent.data.label; 
				counts_map[curr_cat] === undefined ? counts_map[curr_cat] = 1 : counts_map[curr_cat] += 1;
			}			
		}
	});

	var max_count = 0;
	var max_cat = "";
	for (var c in counts_map) {
		if (counts_map[c] > max_count) {
			max_count = counts_map[c];
			max_cat = c;
		}
	}

	insertCategoryIntoLabel(data, cat, max_cat, s);
}

function implementRules(rules) {
	data = initial_data();

	rules.map( function(rule, i) {
		rt = d3.hierarchy(data).sum(function(d){return d.value});
		var s_cat = rule.text.split("-->");
		if (s_cat.length === 2) {
			insertCategoryIntoData(rt, s_cat[1].trim(), s_cat[0].trim(), data);
		}
	});

	redrawSunburst(chart_group, data, arc, click);
}



var initial_data = function() {
	return {
		label: "all spending",
		children: [
			{label: "asda", value: 23},
			{label: "my asda", value: 7},
			{label: "train", value: 32},
			{label: "driving", value: 17},
			{label: "cafe0", value: 13},
			{label: "cafe1", value: 6},
			{label: "driving", value: 43},
			{label: "movie guy", value: 41},		
			{label: "games", value: 10},
			{label: "movies", value: 8},
			{label: "fancy cafe", value: 3},
			{label: "other movie0", value: 5},
			{label: "other movie1", value: 5},
			{label: "other movie2", value: 5}
		]
	};
}

data = initial_data();
rules_data = { 
	rules: [
		{id: 1, text: ""}  
	]
}

Vue.component("special-input", {
	props: ["rule_id", "rule"],
	template: 
		"<div>"+  
	  		"<input v-bind:value='rule'"+
	  		" 		v-on:input='sendRuleUpdate($event)'>"+
	  		"<button v-on:click='close'> X </button>"+   
	  	"</div>",
	methods: {
		sendRuleUpdate: function(e) {
			var rule = e.target.value;
			this.$emit("ruleupdate", this.rule_id, rule);
		},
		close: function() {
			this.$emit("closeit", this.rule_id);
		}
	}
})

simple_input = new Vue({
	el: "#simple-input",
	data: rules_data,
	methods: {
		updateRule: function(rule_id, rule) {
			var ind = getIndex(this.rules, rule_id);
			this.rules[ind].text = rule;
			this.rules = haveOneEmpty(this.rules);
			implementRules(rules_data.rules);
		},
		deleteRule: function(rule_id) {
			this.rules.splice(getIndex(this.rules, rule_id), 1);
			this.rules = haveOneEmpty(this.rules);
			implementRules(rules_data.rules);
		}
	}
});

