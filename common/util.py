def make_task_schedule(objective, n_epochs):
    n_epochs += 1
    if objective == "abuse":
        task_schedule = [("abuse", "abuse", "abuse")]*n_epochs
    elif objective == "link":
        task_schedule = [("link", "link", "link")]*n_epochs
    elif objective == "pretrain-link":
        task_schedule = [("link", "link", "link")]*(19) + \
            [("abuse", "abuse", "abuse")]*(n_epochs)
    elif objective == "pretrain-time":
        task_schedule = [("time", "time", "time")]*(n_epochs-1) + \
            [("abuse", "abuse", "abuse")]*(n_epochs)
    elif objective == "pretrain-triple":
        task_schedule = [("triple", "triple", "triple")]*(n_epochs-1) + \
            [("abuse", "abuse", "abuse")]*(n_epochs)
    elif objective == "multitask":
        task_schedule = []
        for i in range(n_epochs):
            task_schedule.append(("abuse", "abuse", "abuse") if i % 2 == 0 else
                ("link", "abuse", "abuse"))
    return task_schedule
