from flask import Flask, render_template, request, make_response
import random
from ant_colony import AntColony

app = Flask(__name__)

def load_cities_from_file(filepath):
    cities = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        node_coord_section = False
        for line in lines:
            if line.startswith("NODE_COORD_SECTION"):
                node_coord_section = True
                continue
            if line.strip() == "EOF":
                break
            if node_coord_section:
                _, x, y = line.split()
                cities.append((float(x), float(y)))
    return cities

@app.route("/", methods=["GET", "POST"])
def index():
    plot_div = None  # Initialize plot_div
    best_distance = None
    iteration_distances = []

    # Retrieve values from cookies, or use defaults if not set
    num_ants = int(request.cookies.get("num_ants", 25))
    num_iterations = int(request.cookies.get("num_iterations", 100))
    alpha = float(request.cookies.get("alpha", 1.0))
    beta = float(request.cookies.get("beta", 3.0))
    evaporation_rate = float(request.cookies.get("evaporation_rate", 0.1))
    Q = float(request.cookies.get("Q", 100))

    if request.method == "POST":
        # Get values from form and update cookies
        num_ants = int(request.form.get("num_ants", num_ants))
        num_iterations = int(request.form.get("num_iterations", num_iterations))
        alpha = float(request.form.get("alpha", alpha))
        beta = float(request.form.get("beta", beta))
        evaporation_rate = float(request.form.get("evaporation_rate", evaporation_rate))
        Q = float(request.form.get("Q", Q))
        
        # Get the selected TSP file (either "att48" or "berlin52")
        tsp_file = request.form.get("tsp_file", "att48")  # Default to "att48"

        # Create a response to set cookies
        response = make_response(render_template("index.html", plot_div=plot_div, best_distance=best_distance,
                                                 num_ants=num_ants, num_iterations=num_iterations,
                                                 alpha=alpha, beta=beta, evaporation_rate=evaporation_rate, Q=Q, iteration_distances=iteration_distances))

        # Set cookies
        response.set_cookie("num_ants", str(num_ants))
        response.set_cookie("num_iterations", str(num_iterations))
        response.set_cookie("alpha", str(alpha))
        response.set_cookie("beta", str(beta))
        response.set_cookie("evaporation_rate", str(evaporation_rate))
        response.set_cookie("Q", str(Q))

        # Determine the file to load based on the user's selection
        if tsp_file == "att48":
            cities = load_cities_from_file("att48.tsp")
        elif tsp_file == "berlin52":
            cities = load_cities_from_file("berlin52.tsp")
        else:
            cities = load_cities_from_file("att48.tsp")  # Default to att48 if no valid choice

        # Run ant colony optimization
        ant_colony = AntColony(cities, num_ants, num_iterations, alpha, beta, evaporation_rate, Q)
        ant_colony.run()
        plot_div = ant_colony.plot_best_path()  # Get the Plotly div
        best_distance = ant_colony.best_distance
        iteration_distances = ant_colony.iteration_distances
        
        # Update response with new values
        response = make_response(render_template("index.html", plot_div=plot_div, best_distance=best_distance,
                                                 num_ants=num_ants, num_iterations=num_iterations,
                                                 alpha=alpha, beta=beta, evaporation_rate=evaporation_rate, Q=Q, iteration_distances=iteration_distances))
        return response

    # Render the template for GET requests, passing the cookie values
    return render_template("index.html", plot_div=plot_div, best_distance=best_distance,
                           num_ants=num_ants, num_iterations=num_iterations,
                           alpha=alpha, beta=beta, evaporation_rate=evaporation_rate, Q=Q, iteration_distances=iteration_distances)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
