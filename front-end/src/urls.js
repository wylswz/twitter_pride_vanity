import ContentIndex from './Components/Contents/ContentIndex';
import Tweets from './Components/Contents/Tweets'

const ContentRoutes = [
    {
        path: '/',
        component: ContentIndex,
        routes: []
    },
    {
        path: '/tweets',
        component: Tweets,
        routes: []
    }
];

export  {ContentRoutes};