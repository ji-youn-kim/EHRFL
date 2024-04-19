import torch

def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if (args.algorithm == "fedbn" or args.algorithm == "fedpxn"):
            for key in server_model.state_dict().keys():
                if "norm" not in key:
                    temp = torch.zeros_like(
                        server_model.state_dict()[key], dtype=torch.float32
                    )

                    for data in client_weights.keys():
                        temp += (
                            client_weights[data]
                            * models[data].state_dict()[key]
                        )
                    server_model.state_dict()[key].data.copy_(temp)

                    for data in client_weights.keys():
                        models[data].state_dict()[key].data.copy_(
                            server_model.state_dict()[key]
                        )

        else:  # fedavg, fedprox
            for key in server_model.state_dict().keys():
                temp = torch.zeros_like(server_model.state_dict()[key])

                for data in client_weights.keys():
                    temp += (
                        client_weights[data]
                        * models[data].state_dict()[key]
                    )

                server_model.state_dict()[key].data.copy_(temp)

                for data in client_weights.keys():
                    models[data].state_dict()[key].data.copy_(
                        server_model.state_dict()[key]
                    )

    return server_model, models
