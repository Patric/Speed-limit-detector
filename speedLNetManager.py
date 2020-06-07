from speedLNet import *

class speedLNetManager:
    net = 0
    cuda = 0 
    batchsize = 0
    classes = 0
    testloader = 0
    trainloader = 0

    def __init__(self, batchsize):
        self.net = speedLNet()
        self.batchsize = batchsize
        self.cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def imshow(self, img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)))
        plt.show()
       
    def randomFromLoader(self, loader, classes):
        dataiter = iter(loader)
        images, labels = dataiter.next()
        self.imshow(torchvision.utils.make_grid(images))
        print(' '.join('%5s' % classes[labels[j]] for j in range(self.batch_size)))

    def count_E(self, error_value, channels):
        print(f"E = {math.sqrt(error_value/(self.batchsize*16))}")

    # all, training, testing
    def useMNIST(self, datatype):
        self.classes = ('0','1', '2','3', '4','5','6','7','8','9')
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

        if datatype == "training":
            trainset = torchvision.datasets.MNIST('/MNIST_train', download=True, train=True, transform=transform)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size = self.batchsize, shuffle = True, 
                                                        num_workers = 0)
            return self.trainloader

        elif datatype == "testing":
            testset = torchvision.datasets.MNIST('/MNIST_test', download=True, train=False, transform=transform)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size = self.batchsize, shuffle = True, 
                                                        num_workers = 0)
            return self.testloader


        elif datatype == "all":
            trainset = torchvision.datasets.MNIST('/MNIST_train', download=True, train=True, transform=transform)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size = self.batchsize, shuffle = True, 
                                                        num_workers = 0)
            testset = torchvision.datasets.MNIST('/MNIST_test', download=True, train=False, transform=transform)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size = self.batchsize, shuffle = True, 
                                                        num_workers = 0)

            return (self.trainloader, self.testloader)



    def loadData(self, datatype):
        self.classes = ('0', '3', '4', '7', 'unidentified')

        transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=(32,32), interpolation = 1),
        transforms.ToTensor()])

        if datatype == "training":
            trainset = torchvision.datasets.ImageFolder('training_32/', transform = transform)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size = self.batchsize, shuffle = True, 
                                                        num_workers = 0)
            return self.trainloader

        elif datatype == "testing":
            testset = torchvision.datasets.ImageFolder('test_32/', transform = transform)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size = self.batchsize, shuffle = True, 
                                                        num_workers = 0)
            return self.testloader


        elif datatype == "all":
            trainset = torchvision.datasets.ImageFolder('training_32/', transform = transform)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size = self.batchsize, shuffle = True, 
                                                        num_workers = 0)
            testset = torchvision.datasets.ImageFolder('test_32/', transform = transform)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size = self.batchsize, shuffle = True, 
                                                        num_workers = 0)     

            return (self.trainloader, self.testloader)

    def trainNet(self, batchsize, iterations, PATH):
        self.batchsize = batchsize
       
        self.net.to(self.cuda)
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCELoss()
        #criterion = nn.NLLLoss()
        optimiser = optim.SGD(self.net.parameters(), lr = 0.001, momentum = 0.9)

        #self.useMNIST("training")
        self.loadData("training")
        for epoch in range(iterations):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.cuda), data[1].to(self.cuda)
             
                # zero the parameter gradients
                optimiser.zero_grad()
                # forward + backward + optimise
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimiser.step()

                running_loss += loss.item()
                #print statistics
                if i % 10 == 9: #print every 10 mini batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1))
                    running_loss = 0.0
        
        
        torch.save(self.net.state_dict(), PATH)
        print(f"Finished Training. Saved model to {PATH}")
    
    def loadNet(self, PATH):
 
        self.net.load_state_dict(torch.load(PATH))

    def process_image(self, image):
        width, height = image.size
        image = image.resize((255, int()))

    def predict(self, image):
        image.to(self.cuda)
        output = self.net(image)
        #classes = torch.nn.functional.softmax(output)
        label = torch.argmax(output)
     
        #print(f"Torch max{torch.nn.functional.softmax(output)}")
        #print(f"Torch exp{torch.exp(output)}")
      
        return label


    def evaluate(self, image):
        image = image[np.newaxis,:]
        #image = image[np.newaxis,:]

        # converting to PIL image
        # image = torch.FloatTensor(1, image.shape[1], image.shape[0])
        # image = Ft.to_pil_image(image)
        
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image = image.float()
        top_class = self.predict(image)
        #self.classes = ('0','1', '2','3', '4','5','6','7','8','9')
        self.classes = ('0', '3', '4', '7', 'unidentified')

        #print(f"TOP CLASS: {self.classes[top_class]}")
        # if top_class != 0:
        #     print("The model is in certain that the image has a predicted class of ", self.classes[top_class] )
        return self.classes[top_class]

    def showStats(self):
        self.classes = ('0','1', '2','3', '4','5','6','7','8','9')
        #self.loadData("testing")
        self.useMNIST("testing")
        dataiter = iter(self.testloader)
        images, labels = dataiter.next()

        print("Random comparison: ")

        self.imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s' %  self.classes[labels[j]] for j in range(self.batchsize)))

        #self.net = speedLNet()
        #net.load_state_dict(torch.load(PATH))
        outputs = self.net(images)
        print(torch.max(outputs, 1))
        #Max certainty
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%5s' %  self.classes[predicted[j]] for j in range(self.batchsize)))

        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.testloader:
                
                images, labels = data
                outputs = self.net(images)
                # torch.max returns max, max_indices which is predicted in this case
                _, predicted = torch.max(outputs.data, 1)

                # incrementing number of 
                total += labels.size(0)

                correct += (predicted == labels).sum().item()
        print(f"Accuracy of the network on the {total} test images: {round(100 * correct / total, 3)} %")

        class_correct = list(0. for i in range(len(self.classes)))
        class_total = list(0. for i in range(len(self.classes)))
        # no need for gradient. We just print stats
 
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                #print(F"Predicted: {predicted} and {labels}")
                c = (predicted == labels).squeeze()
                # error since sample number % batchsie != 0
                for i in range(self.batchsize):
                    label = labels[i]
                    class_correct[label] += c.item()
                    class_total[label] += 1
                
        for i in range(len(self.classes)):
            print('Accuracy of %5s : %2d %%' % (
                self.classes[i], 100 * class_correct[i] / class_total[i]))



if __name__ == "__main__":
    batchsize = 1
    iterations = 4
    PATH = './sld_my_net.pth'
    trainer = speedLNetManager(batchsize)
    trainer.trainNet(batchsize, iterations, PATH)
    #trainer.loadNet(PATH)
    #trainer.useMNIST("testing")
    #trainer.showStats()