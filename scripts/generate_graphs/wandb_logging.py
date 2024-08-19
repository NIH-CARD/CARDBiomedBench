import wandb
import pandas as pd

WANDB_PROJECT = "github-actions-wandb"

def wandb_boxplot(data: pd.DataFrame,plt):
    # with wandb.init(entity="syed", project=WANDB_PROJECT) as run:
    wandb.init(project=WANDB_PROJECT)
    #get columns
    cols = list(data.columns)
    sel_col=cols[len(cols)-2:len(cols)]
    # Create a table with the columns to plot
    table = wandb.Table(data=data, columns=sel_col)
    wandb.log({
        sel_col[0]: list(data[sel_col[0]]),
        sel_col[1]: list(data[sel_col[1]]),
    })

    wandb.log({"chart": plt})


    # histogram_FACT1 = wandb.plot.histogram(table, value=sel_col[0], title='Histogram')        
    # histogram_FACT2 = wandb.plot.histogram(table, value=sel_col[1], title='Histogram')        
    # wandb.log({'histogram_1': histogram_FACT1, 
    #             'histogram_2': histogram_FACT2})        
