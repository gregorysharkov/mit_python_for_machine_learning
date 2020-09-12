def get_sum_metrics(predictions, metrics=[]):
    if len(metrics)>0:
        metrics = [predictions]
        
    for i in range(3):
        element = predictions + i
        metrics.append(element)
        
    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric
    
    metrics.clear()
    return sum_metrics


def main():
    print(get_sum_metrics(0)==3)  # Should be (0 + 0) + (0 + 1) + (0 + 2) = 3
    print(get_sum_metrics(1)==6)  # Should be (1 + 0) + (1 + 1) + (1 + 2) = 6
    print(get_sum_metrics(2)==9)  # Should be (2 + 0) + (2 + 1) + (2 + 2) = 9
    #print(get_sum_metrics(3, [lambda x: x])==15)  # Should be (3) + (3 + 0) + (3 + 1) + (3 + 2) = 15
    print(get_sum_metrics(0)==3)  # Should be (0 + 0) + (0 + 1) + (0 + 2) = 3
    print(get_sum_metrics(1)==6)  # Should be (1 + 0) + (1 + 1) + (1 + 2) = 6
    print(get_sum_metrics(2)==9)  # Should be (2 + 0) + (2 + 1) + (2 + 2) = 9

if __name__ == "__main__":
    main()
