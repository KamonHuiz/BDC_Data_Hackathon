from .visualize import route_stops_map, all_routes_map
from .data_loader import list_route_ids
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bus route map generator')
    parser.add_argument('--route', '-r', help='Single route id to render (default: first available)')
    parser.add_argument('--all', action='store_true', help='Render all routes aggregated map')
    parser.add_argument('--inbound', action='store_true', help='Use inbound (reverse) stops')
    parser.add_argument('--max-routes', type=int, help='Limit number of routes when using --all')
    parser.add_argument('--output', '-o', help='Output HTML file name')
    args = parser.parse_args()

    outbound = not args.inbound

    if args.all:
        m = all_routes_map(outbound=outbound, max_routes=args.max_routes)
        out_file = args.output or 'all_routes_map.html'
        m.save(out_file)
        print(f'Saved aggregated map to {out_file}')
    else:
        rid = args.route or list_route_ids()[0]
        m = route_stops_map(rid, outbound=outbound)
        out_file = args.output or f'route_{rid}_map.html'
        m.save(out_file)
        print(f'Saved route map to {out_file} for route {rid}')
