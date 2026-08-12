
import torch
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt




class MLP :
    def __init__ (self, words , block_size = 3, embd_dim = 4,h_dim = 100, seed = 2147483647 ) :
        chlist = sorted( list (  set ( ''.join(words) ) ) )
        self.chlist  = ['.']+ chlist
        # Both maps must be built from self.chlist, the list that contains '.'.
        # Built from the 26-letter chlist instead, stoi['a'] would be 0 -- the
        # same index the padding token uses -- and '.' would have no encoding.
        self.stoi ={ st: i for i , st in enumerate (self.chlist ) }
        self.itos = { i: st for i , st in enumerate (self.chlist ) }
        self.volc_dim = len ( self.chlist )
        self.block_size = block_size
        self.embd_dim = embd_dim
        self.h_dim = h_dim
        self.words = words
        # seed = None means do not seed at all: every run differs, which is what
        # you want when checking that a result is not an artefact of one draw.
        self.seed = seed

    def _gen ( self ) :
        # A fresh Generator per call, so weight initialisation and minibatch
        # sampling draw from independent streams -- changing the number of
        # training steps then cannot change the initial weights.
        # torch accepts generator=None and falls back to the global RNG.
        if self.seed is None :
            return None
        return torch.Generator().manual_seed ( self.seed )

    def makeXY (self, num = None ) :
        X = []
        Y = []
        for ww in self.words[:num] :
            # Reset the context for every word.  Initialised once outside this
            # loop, the first rows of each name carry the tail of the previous
            # one instead of the start token.
            xi = [0] * self.block_size
            ww = ww + '.'
            for cc in ww [:] :
                X.append(xi)
                ci = self.stoi[cc]
                Y.append(ci)

                #print ( ''.join ( self.itos[xx] for xx in xi ), '--->', self.itos[ ci] )
                xi = xi[1:] + [ci]
        self.X = torch.tensor( X)
        self.Y= torch.tensor (Y)
        self.Xdic , self.Ydic = {}, {}
        self.Xdic["all"] = self.X
        self.Ydic["all"] = self.Y


    def build_data (self, train  =0.8, dev =0.1, split_seed = 42):
        # Kept separate from self.seed on purpose: you often want the same data
        # split while varying the initialisation, to see how much of a result is
        # the model and how much is the draw.  split_seed = None shuffles freely.
        X = self.X
        Y = self.Y
        n1 = int (len(Y) * train )
        n2 = int ( len (Y) * ( dev + train ) )
        ind = list (range ( len (Y) ) )
        if split_seed is not None :
            random.seed ( split_seed )
        random.shuffle ( ind)
        Xnew = X[ind]
        Ynew = Y[ind]

        self.Xdic["train"] = Xnew [ :n1 ]
        self.Ydic["train"] = Ynew [:n1]
        self.Xdic["dev"] = Xnew [n1:n2]
        self.Ydic ["dev"] = Ynew [n1:n2]
        self.Xdic["test"] = Xnew [n2:]
        self.Ydic["test"] = Ynew [n2:]

        #return Xtr, Ytr, Xdev, Ydev, Xte, Yte


    def init_params ( self, ) :
        g = self._gen()
        # Plain randn, deliberately unscaled.  makemore 3 opens by showing why
        # this is wrong and what to divide by.
        self.C = torch.randn ( ( self.volc_dim , self.embd_dim ), generator = g )
        self.W1 = torch.randn ( ( self.block_size * self.embd_dim , self.h_dim ), generator = g )
        self.b1 = torch.randn ( self.h_dim, generator = g )
        self.W2 = torch.randn ( (  self.h_dim , self.volc_dim), generator = g )
        self.b2 = torch.randn ( self.volc_dim, generator = g )

        self.params = [ self.C, self.W1, self.b1 , self.W2, self.b2]
        for p in self.params :
            p.requires_grad = True


    def forward ( self, Xbatch ) :
        # One forward pass, shared by training and evaluation, so the two can
        # never drift apart.
        emb = self.C[Xbatch].view( ( -1, self.block_size * self.embd_dim  ))
        hact = ( emb@self.W1 + self.b1 ).tanh()
        return hact@self.W2 + self.b2


    def evo_loss ( self, data ='train', numRun = 10000 , minb_size = 32, lr = 0.1)  :

        Xdata = self.Xdic[data]
        Ydata = self.Ydic [data]

        g = self._gen()

        lossi = []
        self.lr = lr

        for ii in range ( numRun )  :

            # mini batch
            ix = torch.randint ( 0, len ( Ydata ) , (minb_size, ), generator = g )
            Xbatch = Xdata[ix]
            Ybatch = Ydata[ix]

            logits = self.forward ( Xbatch )
            loss = F.cross_entropy ( logits, Ybatch)

            # backward
            for p in self.params :
                p.grad = None
            loss.backward()


            # update
            for p in self.params :
                p.data += - lr * p.grad
            lossi.append ( loss.item() )
        self.lossi = lossi
        #print ("loss is ", loss.item() )
        return lossi


    @torch.no_grad()
    def evaluate ( self, data = 'dev' ) :
        # Loss on a whole split, gradients off.  The training loss says whether
        # the model fits; this is the number that says whether it generalises.
        logits = self.forward ( self.Xdic[data] )
        return F.cross_entropy ( logits, self.Ydic[data] ).item()


    def pred_word ( self ) :
        context = [0]* self.block_size
        ww =[]
        while True :
            logits = self.forward ( context )
            prob = torch.softmax ( logits ,dim = -1) 
            xi =torch.multinomial ( prob, num_samples =1 , replacement=False )
            
            context = context [1:] + [xi.item()]
            ww = ww+ [self.itos[xi.item()] ]
            if xi  ==0  :
                break

        word = ''.join ( ww ) 
        return word

if __name__ == "__main__":
    # A smoke test only -- see example_MLP.py for the full run with baselines,
    # diagnostics and the loss figure.
    words = open ( "names.txt", 'r').read().splitlines()

    myclass = MLP ( words, seed = 2147483647 )
    myclass.makeXY()
    myclass.build_data()
    myclass.init_params()
    myclass.evo_loss ( data ='train', numRun = 1000 )


    print ( f"train loss {myclass.evaluate('train'):.4f}" )
    print ( f"dev   loss {myclass.evaluate('dev'):.4f}" )

    ww =myclass.pred_word()
    
    print ( ww)
