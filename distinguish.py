import re
import sys
import time
import argparse
import logging

class State:
    def __init__(self, label, transitions_out, transitions_in):
        self.label = label
        self.tout = transitions_out
        self.tin = transitions_in
    
    def targets(self, action):
        return [trans[2] for trans in self.tout if trans[1] == action]

    def inversetargets(self, action):
        return [trans[0] for trans in self.tin if trans[1] == action]
    
    def splpairs(self): 
        return {(trans[1], trans[2]) for trans in self.tout}

class Formula:
    def __init__(self, modality, conjuncts):
        self.obs_depth = (len(modality) != 0) + max({phi.obs_depth for phi in conjuncts}, default=0)
        self.neg_depth = (len(modality) !=0 and modality[0] == "!") + max({phi.neg_depth for phi in conjuncts}, default=0)
        self.conjuncts = conjuncts
        self.modality = [modality, "true"][modality == ""]
        self.neg_depth = (len(modality) > 0 and modality[0] == "!") + max({phi.neg_depth for phi in conjuncts}, default=0)
        self.size = (len(modality) != 0) + sum({phi.size for phi in conjuncts})

    def __str__(self):
        if len(self.conjuncts) ==0:
            return [self.modality, "true"][len(self.modality) == 0]
        if len(self.conjuncts) > 1:
            return self.modality + "(" + " && ".join([str(phi) for phi in self.conjuncts]) +")"    
        return self.modality +  " ".join([str(phi) for phi in self.conjuncts])

class Block: 
    def __init__(self, parent, leader, lvl,  cleaveland=False, able=False):
        self.parent = parent
        self.lvl = lvl
        self.leader = leader
        self.history = parent.history[:] # slice to copy
        if not cleaveland:
            if lvl > self.parent.lvl:
                for i in range(len(self.history), lvl):
                    self.history.append(self.history[-1])
                    parent.history.append(self.history[-1])
                self.history.append(self)
            else:
                self.history[-1] = self
        else:
            self.history.append(self)
            self.lvl = len(self.history) - 1
        #DATA for cleaveland splitting of type (a, Block)
        self.splpair = None
        self.able = able

    @classmethod
    def root(cls, lts):
        obj = cls.__new__(cls)  # Does not call __init__
        super(Block, obj).__init__() 
        obj.lts = lts
        obj.parent = None
        obj.leader = State(0, [],[])
        obj.lvl = 0
        obj.history = [obj]
        return obj

    def blockatlevel(self, level):
        if level < len(self.history):
            return self.history[level]
        return self 
    
    def states(self):
        return self.lts.partition[self.index]
            
    def __str__(self):
        return "block leader " +  str(self.leader.label) + " op lvl:" + str(self.lvl)

class LTS:
    def __init__(self, transitions, N):
        self.transitions = transitions
        self.N = N
        self.M = len(transitions)
        self.alph = {str(trans[1]) for trans in transitions}
        self.states= [State(i, [], []) for i in range(N)]
        for trans in transitions:
            self.states[trans[0]].tout.append([self.states[trans[0]], trans[1], self.states[trans[2]]])
            self.states[trans[2]].tin.append([self.states[trans[0]], trans[1], self.states[trans[2]]])
        B0 = Block.root(self)
        self.partition = {B0: {self.states[i] for i in range(N)}}
        self.blocks = [B0] * N
        self.wait = [B0]

    def refine(self, i): 
        partition_old = self.partition.copy()
        while self.wait: 
            C = self.wait.pop()
            Cset = partition_old[C]
            alph = {t[1] for s in Cset for t in s.tin}
            for a in alph: 
                Cinv = {t[0] for s in Cset for t in s.tin if t[1] == a}
                self.split(Cinv, i)
        # logging.info("Partition {} has: {} classes".format(i, len(self.partition.keys())))
        for B in self.partition: 
            if B not in partition_old:
                self.wait.append(B)

    def solve(self): 
        i = 1
        while self.wait:
            self.refine(i)
            i += 1
        maxlvl = i - 1 # we did not refine last lvl
        #Make sure every history is equal length
        for B in self.partition:
            while len(B.history) < maxlvl:
                B.history.append(B)
        
    def split(self, Cinv, i):
        blockssplit = {self.blocks[s.label] for s in Cinv}
        for B in blockssplit:
            B1set = self.partition[B].intersection(Cinv)
            if len(B1set) != len(self.partition[B]):
                B2set = self.partition.pop(B).difference(B1set)
                B1 = Block(B,list(B1set)[0], i)
                B2 = Block(B,list(B2set)[0], i)
                self.partition[B1] = B1set
                self.partition[B2] = B2set
                for s in B1set:
                    self.blocks[s.label] = B1
                for s in B2set:
                    self.blocks[s.label] = B2
    
    #split for cleaveland
    def csplit(self, C, a, B, B1set):
        B2set = self.partition.pop(B).difference(B1set)
        B.splpair= (a,C)
        B1 = Block(B,list(B1set)[0], 0, cleaveland=True, able=True)
        B2 = Block(B,list(B2set)[0], 0, cleaveland=True, able=False)
        self.partition[B1] = B1set
        self.partition[B2] = B2set
        for s in B1set:
            self.blocks[s.label] = B1
        for s in B2set:
            self.blocks[s.label] = B2
        self.wait.append(B1)
        self.wait.append(B2)
        if B in self.wait:
            self.wait.remove(B)

    #Paritition refine to get a cleaveland like tree structure
    def cleaveland_solve(self):
        step = 1
        oldlen = 0
        while self.wait: 
            split = False
            #Splitter selection.
            C = self.wait[-1]
            Cset = self.partition[C]
            inversetrans = {(t[0], t[1]) for s in Cset for t in s.tin}
            blockssplit = {self.blocks[t.label] for t, _ in inversetrans}
            for B in blockssplit:
                BCtrans = {(t,ap) for (t,ap) in inversetrans if self.blocks[t.label] == B}
                for a in {ap for (_ ,ap) in BCtrans}:
                    B1set = self.partition[B].intersection({t for (t,ap) in BCtrans if ap == a})
                    if len(B1set) != len(self.partition[B]):
                        self.csplit(C, a, B, B1set)
                        split = True
                        break
            if not split:
                self.wait.remove(C)
            if len(self.partition) > oldlen + step: 
                if (len(self.partition))/step  > 10:
                    step *= 10
                oldlen =len(self.partition)

class CleavelandTree:
    def __init__(self, lts):
        self.lts = lts

    def dist(self, s,t):
        joinB = self.joinblock(s,t)
        (splitact,B) = joinB.splpair
        switch = not self.lts.blocks[s.label].blockatlevel(joinB.lvl+1).able 
        if switch:
            tmp = s
            s = t
            t = tmp    
        ds = {trans[2] for trans in s.tout if trans[1] == splitact and self.lts.blocks[trans[2].label].blockatlevel(B.lvl) == B}
        dt = {trans[2] for trans in t.tout if trans[1] == splitact}
        if len(ds) == 0:
            print(switch)
            exit("Error: Our data structure said there would be a splitting observation, but there was not.")
        for sprime in ds:
            break
        formulas, truths = self.distset(sprime, dt)
        modality = ["<{}>", "!<{}>"][switch].format(splitact)
        formula = Formula(modality, formulas)
        #Calc truths for all blocks.
        newtruths = set()
        for B in self.lts.partition:
            for dt in B.leader.targets(splitact):
                if self.lts.blocks[dt.label] in truths:
                    newtruths.add(B)
                    break
        if switch:
            newtruths = {B for B in self.lts.partition if B not in newtruths}
        return formula, newtruths
    
    def distset(self, s, dt):
        if len(dt) == 0:
            return [], self.lts.partition.keys()
        Gamma = []
        dt_og = dt.copy()
        while len(dt) != 0:
            for t in dt: 
                break
            formula, truths = self.dist(s,t)
            Gamma += [(formula, truths)]
            dt = {t for t in dt if self.lts.blocks[t.label] in truths}
        #Post process:
        Gamma_og = Gamma.copy()
        for (f,T) in Gamma_og:
            newTruths =  set.intersection(set(self.lts.partition.keys()), *[dT for (_, dT) in Gamma if dT != T])  
            if not {t for t in dt_og if self.lts.blocks[t.label] in newTruths}:
                # logging.info("Intermediate conj. removal. {}".format(len(Gamma)))
                Gamma.remove((f,T))
        Gamma1 = [g[0] for g in Gamma]
        Truths = [g[1] for g in Gamma]      
        return Gamma1, Truths[0].intersection(*Truths)

    def joinblock(self, s,t):
        B1 = self.lts.blocks[s.label]
        B2 = self.lts.blocks[t.label]
        maxlvl = min(len(B1.history), len(B2.history))
        for i in range(maxlvl):
            if B1.history[i] != B2.history[i]:
                return B1.history[i-1]
        exit("Error: We appear to trying to distinguish states that are bisimilar.")

    def intersect_level(self, B1,B2):
        maxlvl = min(len(B1.history), len(B2.history))
        for i in range(maxlvl):
            if B1.history[i] != B2.history[i]:
                return i

class DistTree:
    def __init__(self, lts):
        self.lts = lts
        maxlvl = len(self.lts.blocks[0].history)
        self.historyTree = {i: set() for i in range(maxlvl)}
        for B in self.lts.partition:
            for i in range(maxlvl):
                self.historyTree[i].add(B.history[i]) 
        self.cached_dirdist = dict()

    def lifttruths(self,truths, lvl, goal):
        if goal < lvl+1:
            sys.exit("ERR01: can't lift truths downwards")
        if goal > len(self.historyTree)-1:
            sys.exit("ERR02: Cant lift truths beyond maxlvls")
        if lvl == goal:
            return truths    
        return {B for B in self.historyTree[goal] if B.history[lvl] in truths}

    # Formula such that B1 in and B2 not in
    def dist(self, B1, B2):
        lvl = self.intersect_level(B1, B2)
        if B1.history[lvl-1] != B2.history[lvl-1]:
            sys.exit("Whoops!")
        ds = {(a, self.lts.blocks[s.label].history[lvl-1]) for (a,s) in B1.leader.splpairs()}
        dt = {(a, self.lts.blocks[t.label].history[lvl-1]) for (a,t) in B2.leader.splpairs()}
        deltai = {(a,s) for (a,s) in ds if (a,s) not in dt}
        if(len(deltai)==0):
            # we should negate dist(b2, b1)
            formula, truths = self.dist(B2,B1)
            return (Formula("!"+formula.modality, formula.conjuncts), {B for B in self.historyTree[lvl] if B not in truths})
        for (splitact,sprime) in deltai:
            break
        targets = {t for (a,t) in dt if a == splitact}
        formulas, truths =  self.distset(sprime, targets, lvl-1)
        #Lift the truths one lvl, return the combo.
        newformula = Formula("<{}>".format(splitact), formulas)
        newtruths = set() 
        for B in self.historyTree[lvl]:
            for dt in B.leader.targets(splitact):
                if self.lts.blocks[dt.label].history[lvl-1] in truths:
                    newtruths.add(B)
                    break
        return newformula, newtruths

    def distset(self, block, dt, lvl):
        if len(dt) == 0:
            return ("", self.historyTree[lvl])
        Gamma = []
        while len(dt) != 0:
            lvls = [(t, self.intersect_level(block,t)) for t in dt]
            #FLAG: slow
            t, distlvl = max(lvls, key=lambda x: x[1])
            # Gen formula for this derivative
            formula, truths = self.dist(block ,t)
            if distlvl < lvl: 
                truths = self.lifttruths(truths,distlvl,lvl)
                #lift truths to this lvls which is lvl 
            Gamma += [(formula, truths)]
            dt = dt.intersection(truths)
        Gamma1 = [g[0] for g in Gamma]
        Truths = [g[1] for g in Gamma]
        if len(Gamma) > 1:
            return ( Gamma1, Truths[0].intersection(*Truths))
        return ( Gamma1, Truths[0].intersection(*Truths))

    def intersect_level(self, B1,B2):
        for i in range(len(self.historyTree)):
            if B1.history[i] != B2.history[i]:
                return i
        sys.exit("Error #13: We try to distinguish blocks that seem to be bisimilar")

    def dirdistsingle(self, B0, B1, lvl=-1):
        if (B0, B1) in self.cached_dirdist:
            return self.cached_dirdist[(B0,B1)]
        self.cached_dirdist[(B0, B1)] = -1 
        i = [self.intersect_level(B0, B1), lvl][lvl!=-1]
        ds = {(a, self.lts.blocks[s.label].blockatlevel(i-1)) for (a,s) in B0.leader.splpairs()}
        dt = {(a, self.lts.blocks[t.label].blockatlevel(i-1)) for (a,t) in B1.leader.splpairs()}
        deltai = {(a,s) for (a,s) in ds if (a,s) not in dt}
        candidates = []
        for (splitact,sprime) in deltai:
            targets = {t for (a,t) in dt if a == splitact}
            candidates.append(max([0] + [self.dirdistsingle(sprime, t, lvl=i-1) for t in targets]))
        deltait = {(a,t) for (a,t) in dt if (a,t) not in ds}
        for (at,tprime) in deltait:
            targets = {s for (a,s) in ds if a == at} 
            candidates.append(max([1] + [self.dirdistsingle(tprime, sp, lvl=i-1) + 1 for sp in targets]))
        self.cached_dirdist[(B0,B1)] = min(candidates)
        return min(candidates)


class AnnotatedFormula:
    def __init__(self,formula, lts, nr):
        self.nr = nr
        self.maxnr = nr
        self.modality = formula.modality
        self.conjuncts = []
        for c in formula.conjuncts:
            newFormula = AnnotatedFormula(c, lts, self.maxnr+1)
            self.maxnr = newFormula.maxnr
            self.conjuncts.append(newFormula)
        self.lts = lts
        self.allblocks = set(lts.partition.keys())
        self.cached_truths = set()
        self.formula = formula

    def truths(self, recompute = False):
        if len(self.cached_truths) == 0 or recompute:
            if self.modality == "true":
                return self.allblocks
            if not self.conjuncts:
                truths = self.allblocks
            else:
                truths = set.intersection(*[C.truths(recompute) for C in self.conjuncts])
            a = [self.modality[1:-1], self.modality[2:-1]][self.modality[0]=="!"]
            if self.modality[0] == "!":
                self.cached_truths = {B for B in self.allblocks if not [sprime for sprime in B.leader.targets(a) if self.lts.blocks[sprime.label] in truths]}
            else:  
                self.cached_truths = {B for B in self.allblocks if [sprime for sprime in B.leader.targets(a) if self.lts.blocks[sprime.label] in truths]}
        return self.cached_truths
    
    def truths_without(self, filter):
        if filter == self.nr: 
            return self.allblocks
        if filter <= self.maxnr and filter > self.nr and self.conjuncts:
            truths = set.intersection(*[C.truths_without(filter) for C in self.conjuncts])
            a = [self.modality[1:-1], self.modality[2:-1]][self.modality[0]=="!"]
            if self.modality[0] == "!":
                return {B for B in self.allblocks if not [sprime for sprime in B.leader.targets(a) if self.lts.blocks[sprime.label] in truths]}
            else:  
                return {B for B in self.allblocks if [sprime for sprime in B.leader.targets(a) if self.lts.blocks[sprime.label] in truths]}
        return self.truths()
    
    def remove(self, filter): 
        if self.nr == filter:
            self.modality = "true"
            self.conjuncts = []
            self.cached_truths = set()
            return self.maxnr
        else: 
            for c in self.conjuncts:
                if c.maxnr >= filter and c.nr <= filter: 
                    return c.remove(filter)

    def to_formula(self): 
        return Formula([self.modality, ""][self.modality =="true"], [c.to_formula() for c in self.conjuncts])
    
    def __str__(self):
        if len(self.conjuncts) ==0:
            if self.modality == "true": 
                return self.modality
            return self.modality + "true"
        if len(self.conjuncts) > 1:
            return self.modality + "(" + " && ".join([str(phi) for phi in self.conjuncts]) +")"    
        return self.modality +  " ".join([str(phi) for phi in self.conjuncts])


def readLTS(filename, offset=0):
    file = open(filename,"r")
    lines= file.read().split("\n")
    file.close()
    infoline = re.match(r'.*[(](.*)[)]', lines[0])
    S0,M,N = [int(s) for s in infoline.groups()[0].split(",")]
    transitions = []
    if M != len(lines[1:-1]):
        sys.exit("Miss match in {}, M={}, but found {} transitions".format(filename, M,len(lines[1:-1])))

    for line in lines[1:-1]:
        split = line[1:-1].split('"')
        source, a, target = split[0][:-1],  split[1], split[2][1:]
        transitions.append([int(source)+offset, a, int(target)+offset])
    return (S0+offset, N, transitions)

def postProcess(formula, lts, start0, start1):
    i = 0
    B0, B1 = lts.blocks[start0], lts.blocks[start1]
    aFormula = AnnotatedFormula(formula, lts, i)
    Gamma = aFormula.truths()
    if B0 not in Gamma or B1 in Gamma:
        exit("This is not supposed to happen, this formula seems not distinguishing")
    logging.info("seeing if we can remove one of the {} conjuncts".format(aFormula.maxnr + 1))
    i = 0
    while i <= aFormula.maxnr:
        nt = aFormula.truths_without(i)
        if B0 in nt and B1 not in nt:
            logging.info("We found a conjunct to remove!")
            i = aFormula.remove(i) 
            Gamma = aFormula.truths(True)
            if B0 not in Gamma or B1 in Gamma:
                print("ERRNO")
                exit("This is not supposed to happen, this formula seems not distinguishing")
        i += 1
    return aFormula.to_formula()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'LTSdist',
                    description = 'Compute a distinguishing HML formula')
    parser.add_argument('infile1', help='Input LTS 1, input should be Aldebaran format (.aut)')
    parser.add_argument('infile2', help= 'Input LTS 2, input should be Aldebaran format (.aut)')
    parser.add_argument('-c', '--cleaveland', action='store_true')    
    parser.add_argument('-v', '--verbose', action='store_true') 
    parser.add_argument('-b', '--benchmark', action='store_true', help="only output the metrics of the distinguishing formula.") 
    parser.add_argument('-p', '--postprocess', action='store_true')    
    parser.add_argument('--logfile', help="logfile to write to")    

    args = parser.parse_args()
    file1 = args.infile1 #File 1
    file2 = args.infile2 #File 2
    logfile = [file2.split("\\")[-1], args.logfile][args.logfile is not None]
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    if args.logfile is not None:
        logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO , filename='{}.log'.format(logfile))

    start0, N1, trans1 = readLTS(file1)
    start1, N2, trans2 = readLTS(file2, offset = N1)
    L = LTS(trans1 +trans2, N1 + N2)
    st = time.time()
    sys. setrecursionlimit(50000) 
    logging.info("Starting an iteration comparing {} {}...".format(file1, file2))
    if args.cleaveland:
        logging.info("Solving bisimilarity suitable for cleaveland (might be slower)...")
        L.cleaveland_solve()
    else:
        logging.info("Solving k-bisimilarity...")
        L.solve()
    et = time.time()
    elapsed_time = et - st
    logging.info("We solved bisimilarity... in {} seconds and  {} classes and depth {}".format(elapsed_time, len(L.partition), len(L.blocks[0].history)-1))
    if args.cleaveland:
        distTree = CleavelandTree(L)
        formula, _ = distTree.dist(L.blocks[start0].leader,L.blocks[start1].leader)
    else:
        distTree = DistTree(L) 
        formula, _ = distTree.dist(L.blocks[start0], L.blocks[start1])
    et2 = time.time()
    elapsed_time = et2 - et
    logging.info("Formula computed in {} seconds".format(elapsed_time))
    logging.info("Formula has depth:{}, size: {}, negdepth:{}".format(formula.obs_depth, formula.size, formula.neg_depth))
    if not args.benchmark:
        print(formula)
    else: 
        logging.info("Formula:{}".format(str(formula)))
        print("Metrics({};{};{})".format(formula.obs_depth,formula.size, formula.neg_depth))

    if args.postprocess:
        updated_formula = postProcess(formula, L, start0, start1)
        et3 = time.time()
        logging.info("Postprocessing took {} seconds".format(et3 - et2))
        if not args.cleaveland:
            logging.info("Minimal number of negations no depth:{}".format(distTree.dirdistsingle(L.blocks[start0], L.blocks[start1], len(L.partition))))
        logging.info("uFormula:{}".format(updated_formula))
        logging.info("Formula has depth:{}, size: {}, negdepth:{}".format(updated_formula.obs_depth, updated_formula.size, updated_formula.neg_depth))
        if not args.benchmark:
            print(updated_formula)
        else:
            print("MetricsAfterClean({};{};{})".format(updated_formula.obs_depth,updated_formula.size, updated_formula.neg_depth))
    sys.exit()

# Check the generated mcf formula with 
# lts2pbes.exe --lps=1394-fin.lps -p --formula=1394-mut.mcf .\1394-fin.aut 1394-fin-orig.pbes
