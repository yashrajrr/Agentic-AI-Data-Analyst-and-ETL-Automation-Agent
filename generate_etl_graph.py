from agents.etl_analyst import etl_analyst; img = etl_analyst.get_graph().draw_mermaid_png(); open('etl_analyst_graph.png', 'wb').write(img)
