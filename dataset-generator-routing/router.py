import random

ROUTE_WEIGHTS = {
    "R1": 0.35,
    "R2": 0.25,
    "R3": 0.25,
    "R4": 0.15,
}

def get_eligible_routes(transaction, routes):

    eligible = []

    for route in routes:
        payment_supported = (transaction.payment_method in route.supported_payment_methods)
        network_supported = (transaction.network in route.supported_networks)

        if payment_supported and network_supported:
            eligible.append(route)

    return eligible

def choose_route(transaction, routes):
    eligible_routes = get_eligible_routes(transaction, routes)

    if not eligible_routes:
        return None

    weights = [ROUTE_WEIGHTS[route.route_id] for route in eligible_routes]

    return random.choices(eligible_routes,weights=weights,k=1)[0]