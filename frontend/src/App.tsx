import { useState } from 'react';
import Sidebar from './components/Sidebar';
import HealthPage from './pages/HealthPage';
import IngestPage from './pages/IngestPage';
import SearchPage from './pages/SearchPage';
import AskPage from './pages/AskPage';
import ChatPage from './pages/ChatPage';
import EvalPage from './pages/EvalPage';
import SearchAgentPage from './pages/SearchAgentPage';
import AgentPage from './pages/AgentPage';
import LiteraturePage from './pages/LiteraturePage';
import CaseSearchPage from './pages/CaseSearchPage';
import DataManagePage from './pages/DataManagePage';
import ModelSwitchPage from './pages/ModelSwitchPage';
import ParamsConfigPage from './pages/ParamsConfigPage';
import type { AuthUser } from './types';

const GUEST_USER: AuthUser = {
  user_id: 'guest',
  token: '',
  role: 'patient',
  phone: '游客模式',
};

type PageId =
  | 'health'
  | 'ingest'
  | 'search'
  | 'ask'
  | 'chat'
  | 'eval'
  | 'search-agent'
  | 'agent'
  | 'literature'
  | 'case-search'
  | 'data-manage'
  | 'model-switch'
  | 'params-config';

function PageContent({ page, user }: { page: PageId; user: AuthUser }) {
  switch (page) {
    case 'health': return <HealthPage />;
    case 'ingest': return <IngestPage />;
    case 'search': return <SearchPage />;
    case 'ask': return <AskPage />;
    case 'chat': return <ChatPage user={user} />;
    case 'eval': return <EvalPage />;
    case 'search-agent': return <SearchAgentPage />;
    case 'agent': return <AgentPage user={user} />;
    case 'literature': return <LiteraturePage />;
    case 'case-search': return <CaseSearchPage />;
    case 'data-manage': return <DataManagePage />;
    case 'model-switch': return <ModelSwitchPage />;
    case 'params-config': return <ParamsConfigPage />;
    default: return <AskPage />;
  }
}

export default function App() {
  const [user, setUser] = useState<AuthUser>(() => {
    try {
      const raw = localStorage.getItem('auth_user');
      return raw ? (JSON.parse(raw) as AuthUser) : GUEST_USER;
    } catch {
      return GUEST_USER;
    }
  });

  const [currentPage, setCurrentPage] = useState<PageId>(() => 'chat');

  const handleLogout = () => {
    localStorage.removeItem('auth_user');
    setUser(GUEST_USER);
    setCurrentPage('chat');
  };

  // Chat and agent pages need to fill the full height
  const isFullHeightPage = currentPage === 'chat' || currentPage === 'agent';

  return (
    <div className="flex h-screen bg-slate-50 overflow-hidden">
      <Sidebar
        currentPage={currentPage}
        onNavigate={(id) => setCurrentPage(id as PageId)}
        user={user}
        onLogout={handleLogout}
      />

      <main
        className={`flex-1 overflow-hidden ${
          isFullHeightPage ? 'flex flex-col' : 'overflow-y-auto scrollbar-thin'
        }`}
      >
        {isFullHeightPage ? (
          <div className="flex-1 overflow-hidden flex flex-col">
            <PageContent page={currentPage} user={user} />
          </div>
        ) : (
          <div className="min-h-full">
            <PageContent page={currentPage} user={user} />
          </div>
        )}
      </main>
    </div>
  );
}
